# fVAR algorithm
```r
#before opening R studio,activate the requrired python environment in anaconda prompt
library(reticulate)
library(rsofun)
library(dplyr)
library(readr)
library(lubridate)
library(ggplot2)

```
# Usage example

## Get FLUXNET 2015 data

Training data is to be read in as a data frame with column names corresponding to the variable names defined by `settings` (see below).

Read observational data including soil moisture for one site (see column `sitename`) from FLUXNET 2015 data. The following loads data extracted from original files by `data-raw/prepare_data_example.R`.
```r
load("./data/df_fluxnet.Rdata")
```

## Define settings

This named list is used by several functions of the package.
```r
# ## use all observational soil moisture data
# varnams_soilm <- df_fluxnet %>% 
#   dplyr::select( starts_with("SWC_") ) %>% 
#   dplyr::select( -ends_with("QC") ) %>% 
#   names()

settings <- list( 
  target        = "GPP_NT_VUT_REF", 
  predictors    = c("temp_day","vpd_day", "ppfd", "fpar","wcont_splash"), 
  varnams_soilm = "wcont_splash",
  nnodes        = 10,
  nrep          = 3,
  package       = "nnet"
  )#注意这里的predictors还是给它添加了soilm，因为下一个函数train_predict_fvar马上就用了
```

The settings are a named list that must contain the following variables:

- `target`        : Character, target variable for the NN model, must correspond to the respective column name in the training data.
- `predictors`    : Vector of characters, predictor variables (excluding soil moisture) for the NN model, must correspond to the respective column name in the training data.
- `varnams_soilm` : (Vector of) character(s). Name of the soil moisture variable(s). Must correspond to the respective column name in the training data.
- `nnodes`        : Number of hidden nodes of the NN.
- `nrep`          : Number of repetitions of NN model training. `fvar` is calculated as a mean across repetitions.
- `package`       : R package name that implements the NN (for now, use `"nnet"`, which is used in combination with `caret`)


## Prepare data

To prepare training data, removing NAs and outliers, in a standardized way, do:
```r
library(tidyr)
source("./R/align_events.R")
source("./R/get_consecutive.R")
source("./R/get_droughts_fvar.R")
source("./R/get_obs_bysite_fluxnet2015.R")
source("./R/get_opt_threshold.R")
source("./R/identify_pattern.R")
source("./R/mct.R")
source("./R/plot_fvar_cwd.R")
source("./R/predict_nn.R")
source("./R/predict_nn_keras.R")
source("./R/prepare_trainingdata_fvar.R")
source("./R/profile_soilmthreshold_fvar.R")
source("./R/prune_droughts.R")
source("./R/remove_outliers.R")
source("./R/test_performance_fvar.R")
source("./R/train_predict_fvar.R")#这里先source了所有可能需要用到的函数
df_train <- prepare_trainingdata_fvar( df_fluxnet, settings )
```
#works well until here_Qin
## Train model and predict fvar

Train models and get `fvar` (is returned as one column in the returned data frame). Here, we specify the soil moisture threshold to separate training data into moist and dry days to 0.6 (fraction of water holding capacity). Deriving an optimal threshold is implemented by the functions `profile_soilmthreshold_fvar()` and `get_opt_threshold()` (see below).
```r
library(keras)
#library( tensorflow)
library( nnet )
library( caret )
df_nn_soilm_obs <- train_predict_fvar( 
  df_train,
  settings,
  soilm_threshold = 0.6,
  weights         = NA, 
  verbose         = TRUE
  )#here comes the problem
```

## Evaluate performance

The function `test_performance_fvar()` returns a list of all ggplot objects and a data frame with evaluation results for multiple tests.
```r
testresults_fvar <- test_performance_fvar(df_nn_soilm_obs, settings)
```

## Get optimal soil moisture threshold

The two steps (train/predict and performance evaluation) are executed for a set of soil moisture thresholds. The optimal threshold is then selected based on the different performance criteria.

First, run the fvar algorithm for a set of soil moisture thresholds and record performance metrics for each round.
```r
df_profile_performance <- profile_soilmthreshold_fvar(
  df_train,
  settings,
  weights = NA,    # optional weights
  len = 4          # number of soil moisture thresholds
  )
```

Then select best threshold as described in Stocker et al. (2018). Their procedure was as follows:

1. Determine the difference in the bias of NN$_\text{pot}$ between dry and moist days and select the three (originally used five) thresholds with the greatest difference.
2. Determine RMSE of NNpot vs. NNact during moist days and take the threshold where it's minimised

```r
purrr::map(df_profile_performance, "df_metrics") %>% 
  bind_rows(.id = "threshold") %>% 
  arrange(-diff_ratio_dry_moist) %>% 
  slice(1:3) %>% 
  arrange(rmse_nnpot_nnact_moist) %>% 
  slice(1) %>% 
  pull(threshold) %>% 
  as.numeric()
```

This is implemented by a function:
```r
get_opt_threshold(df_profile_performance)
```

Repeat the fvar algorithm with optimal threshold.
```r
df_nn <- train_predict_fvar( 
  df_train,
  settings,
  soilm_threshold = get_opt_threshold(df_profile_performance),
  weights         = NA
  )
```

## Evaluations and visualisations

### Identify soil moisture droughts

Determine drought events based on fvar.
```r
out_droughts <- get_droughts_fvar( 
  df_nn, 
  nam_target     = settings$target, 
  leng_threshold = 10, 
  df_soilm       = select(df_fluxnet, date, soilm=wcont_swbm),
  df_par         = select(df_fluxnet, date, par=ppfd),
  par_runmed     = 10
  )
```

Plot time series of fvar and highlight drought events.
```r
ggplot() +
  geom_rect(
    data=out_droughts$droughts, 
    aes(xmin=date_start, xmax=date_end, ymin=-99, ymax=99), 
    fill=rgb(0,0,0,0.3), 
    color=NA) + 
  geom_line(data=out_droughts$df_nn, aes(date, fvar_smooth_filled)) +
  coord_cartesian(ylim=c(0, 1.2))
```

### Align data by droughts

After having determined drought events above, "align" data by the day of drought onset and aggregate data across multiple drought instances.
```r
dovars <- c("GPP_NT_VUT_REF", "fvar", "soilm", "nn_pot", "nn_act")
df_nn <- out_droughts$df_nn %>% mutate(site="FR-Pue")
df_alg <- align_events( 
  select( df_nn, site, date, one_of(dovars)), 
  select( df_nn, site, date, isevent=is_drought_byvar_recalc ),
  dovars,
  do_norm=FALSE,
  leng_threshold=10, 
  before=20, after=80, nbins=10
  )
```

Visualise data aggregated by drought events.
```r
df_alg$df_dday_agg_inst_site %>% 
  ggplot(aes(dday, fvar_median)) +
  geom_ribbon(aes(ymin=fvar_q33, ymax=fvar_q66), fill="grey70") +
  geom_line() +
  geom_vline(xintercept=0, linetype="dotted") +
  ylim(0,1) + 
  labs(title = "fVAR")

median <- df_alg$df_dday_agg_inst_site %>%
  select(dday, nn_pot=nn_pot_median, nn_act=nn_act_median, gpp_obs=GPP_NT_VUT_REF_median) %>% 
  tidyr::gather(model, var_nn_median, c(gpp_obs, nn_pot, nn_act))

q33 <- df_alg$df_dday_agg_inst_site %>%
  select(dday, nn_pot=nn_pot_q33, nn_act=nn_act_q33, gpp_obs=GPP_NT_VUT_REF_q33) %>% 
  tidyr::gather(model, var_nn_q33, c(gpp_obs, nn_pot, nn_act))

q66 <- df_alg$df_dday_agg_inst_site %>%
  select(dday, nn_pot=nn_pot_q66, nn_act=nn_act_q66, gpp_obs=GPP_NT_VUT_REF_q66) %>% 
  tidyr::gather(model, var_nn_q66, c(gpp_obs, nn_pot, nn_act))

df_tmp <- median %>% 
  left_join(q33, by=c("dday", "model")) %>% 
  left_join(q66, by=c("dday", "model"))

df_tmp %>% 
  ggplot(aes(dday, var_nn_median, color=model)) +
  geom_line() +
  geom_ribbon(aes(ymin=var_nn_q33, ymax=var_nn_q66, fill=model), alpha=0.3, color=NA) +
  geom_vline(xintercept=0, linetype="dotted") +
  labs(
    title = "GPP", 
    subtitle = "Observation and models",
    x="dday", y="GPP (g C m-2 d-1)") +
  scale_color_discrete(
    name="Model", 
    breaks=c("transp_obs", "nn_act", "nn_pot"),
    labels=c("Observed", expression(NN[act]), expression(NN[pot])) ) +
  scale_fill_discrete(
    name="Model", 
    breaks=c("transp_obs", "nn_act", "nn_pot"),
    labels=c("Observed", expression(NN[act]), expression(NN[pot])) )
```
