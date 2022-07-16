#fVAR algorithm
```r
library(reticulate);
library(rsofun)
library(dplyr)
library(plyr)
library(readr)
library(lubridate)
library(ggplot2)
library(visdat)
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
settings <- list(
  target        = "GPP_NT_VUT_REF", 
  predictors    = c("temp_day","vpd_day", "ppfd", "fpar","wcont_splash"), 
  varnams_soilm = "wcont_splash",
  nnodes_pot    = 15,
  nnodes_act    = 20,
  nrep          = 3,
  package       = "nnet",
  rowid         = "date"
  )#注意这里的predictors还是给它添加了soilm，因为下一个函数train_predict_fvar马上就用了，且增加了date作为行id,pot和act的nnodes弄的不一样，不晓得是为啥
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
#source("./R/plot_fvar_cwd.R")
source("./R/predict_nn.R")
#source("./R/predict_nn_keras.R")
source("./R/prepare_trainingdata_fvar.R")
source("./R/profile_soilmthreshold_fvar.R")
source("./R/prune_droughts.R")
source("./R/remove_outliers.R")
source("./R/test_performance_fvar.R")
source("./R/train_predict_fvar.R")#这里先source了所有可能需要用到的函数
df_train <- prepare_trainingdata_fvar( df_fluxnet, settings )
```
## Train model and predict fvar

Train models and get `fvar` (is returned as one column in the returned data frame). Here, we specify the soil moisture threshold to separate training data into moist and dry days to 0.6 (fraction of water holding capacity). Deriving an optimal threshold is implemented by the functions `profile_soilmthreshold_fvar()` and `get_opt_threshold()` (see below).
```r
# library(tidyverse)
#library(keras)
#library( tensorflow)
# library(reticulate)
#library(tensorflow)
library( nnet )
library( caret )
df_nn_soilm_obs <- train_predict_fvar( 
  df_train,
  settings,
  soilm_threshold = 0.6,
  weights         = NA, 
  verbose         = TRUE
  )
```
```{r}
out <- df_nn_soilm_obs#here we use out becase Francesco used it, to keep consistent
df_all <- df_nn_soilm_obs$df_all
#要读出来才能存
write.csv (df_all,"df_all_2.csv",col.names = NULL, row.names = FALSE)
#将df_all存成csv格式
```
## Evaluate performance

The function `test_performance_fvar()` returns a list of all ggplot objects and a data frame with evaluation results for multiple tests.
```r
testresults_fvar <- test_performance_fvar(out$df_all,settings)
```
#if for details

## 1. NNact has no systematic bias related to the level of soil moisture.
library(LSD)
library (ggthemes)
library (RColorBrewer)
library (rbeni)
#load above library first before you perform following ones
df_test_performance <- df_all %>% 
  mutate(bias_act = nn_act - obs,
         bias_pot = nn_pot - obs,
         soilm_bin = cut(soilm, 10),
         ratio_act = nn_act / obs,
         ratio_pot = nn_pot / obs
  ) 

df_test_performance %>%
  tidyr::pivot_longer(cols = c(bias_act, bias_pot), names_to = "source", values_to = "bias") %>% 
  ggplot(aes(x = soilm_bin, y = bias, fill = source)) +
  geom_boxplot() +
  geom_hline(aes(yintercept = 0.0), linetype = "dotted") +
  labs(title = "Bias vs. soil moisture")
## test whether slope is significantly different from zero (0 is within slope estimate +/- standard error of slope estimate)
linmod <- lm(bias_act ~ soilm, data = df_test_performance)
testsum <- summary(linmod)
slope_mid <- testsum$coefficients["soilm","Estimate"]
slope_se  <- testsum$coefficients["soilm","Std. Error"]
passtest_bias_vs_soilm <- ((slope_mid - slope_se) < 0 && (slope_mid + slope_se) > 0)

## record for output
df_out_performance <- tibble(slope_bias_vs_soilm_act = slope_mid, passtest_bias_vs_soilm = passtest_bias_vs_soilm)

#2. NN$_\text{pot}$ and NN$_\text{act}$ have no bias during moist days.

df_test_performance %>% 
  tidyr::pivot_longer(cols = c(bias_act, bias_pot), names_to = "source", values_to = "bias") %>% 
  dplyr::filter(moist) %>% 
  ggplot(aes(y = bias, fill = source)) +
  geom_boxplot() +
  geom_hline(aes(yintercept = 0), linetype = "dotted")
df_test_performance %>% 
  dplyr::filter(moist) %>%
  summarise(bias_act = mean(bias_act, na.rm = TRUE), bias_pot = mean(bias_pot, na.rm = TRUE))

#3. NN$_\text{pot}$ and NN$_\text{act}$ have a good predictive model performance (high *R*$^2$ and low RMSE measured on cross-validation resamples) during moist days.
## NN act
out_modobs <- out$df_cv %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_act", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN act, moist days, CV")

out_modobs <- out$df_all %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_act", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN act, moist days, all")
## NN pot
out_modobs <- out$df_cv %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_pot", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN pot, moist days, CV")

out_modobs <- out$df_all %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_pot", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN pot, moist days, all")

#4. NN$_\text{act}$ has a good predictive model performance (high *R*$^2$ and low RMSE measured on cross-validation resamples) during dry days.
## based on CV
out_modobs <- out$df_cv %>% 
  dplyr::filter(!moist) %>% 
  rbeni::analyse_modobs2("nn_act", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN act, dry days, cv")#actually here it is almost the same when use all or cv

out_modobs <- out$df_all %>% 
  dplyr::filter(!moist) %>% 
  rbeni::analyse_modobs2("nn_act", "obs", type = "heat")
out_modobs$gg +
  labs(title = "NN act, dry days, all")
#4. NN$_\text{pot}$ and NN$_\text{act}$ have a high agreement (high *R*$^2$ and low RMSE) during moist days.

## based on all
out_modobs <- out$df_all %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_pot", "nn_act", type = "heat")
out_modobs$gg
labs(title = "NN act, moist days, all")# similar to above but better performance than dry days

out_modobs <- out$df_cv %>% 
  dplyr::filter(moist) %>% 
  rbeni::analyse_modobs2("nn_pot", "nn_act", type = "heat")
out_modobs$gg
labs(title = "NN act, moist days, cv")# similar to above but better performance than dry days

## Metrics for use as in Stocker et al. (2018)
## 1. Determine the difference in the bias of NNpot between dry and moist days
df_ratio_moist <- df_test_performance %>% 
  dplyr::filter(moist) %>%
  summarise(ratio_act = median(ratio_act, na.rm = TRUE), ratio_pot = median(ratio_pot, na.rm = TRUE))

df_ratio_dry <- df_test_performance %>% 
  dplyr::filter(!moist) %>%
  summarise(ratio_act = median(ratio_act, na.rm = TRUE), ratio_pot = median(ratio_pot, na.rm = TRUE))

## 2. Determine RMSE of NNpot vs. NNact during moist days
rmse_nnpot_nnact_moist <- df_test_performance %>% 
  dplyr::filter(moist) %>%
  yardstick::rmse(nn_act, nn_pot) %>% 
  pull(.estimate)

df_out_perf <- df_out_performance %>% 
  mutate(
    diff_ratio_dry_moist = df_ratio_dry$ratio_pot - df_ratio_moist$ratio_pot,
    rmse_nnpot_nnact_moist = rmse_nnpot_nnact_moist
  )
library(ggpubr)

#now the evaluation of model performance is completed. the save of local variables did make the results very tedious

## Get optimal soil moisture threshold
df_profile_performance <- profile_soilmthreshold_fvar(
  df_train,
  settings,
  weights = NA,    # optional weights
  len = 4          # number of soil moisture thresholds
)




## Get optimal soil moisture threshold

The two steps (train/predict and performance evaluation) are executed for a set of soil moisture thresholds. The optimal threshold is then selected based on the different performance criteria.

First, run the fvar algorithm for a set of soil moisture thresholds and record performance metrics for each round.
```r
library(dplyr)
library(plyr)
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
