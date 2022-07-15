#' Trains models for potential and actual fluxes. 
#' 
#' @param df A data frame containing observational data for all the predictors and training variables with all NAs removed.
#' @param settings A character string defining which variable (column name in \code{df}) is to be used as target variable.
#' 
train_predict_fvar <- function( df, settings, soilm_threshold, weights=NA, plot=FALSE, verbose = FALSE ){
  
  ##------------------------------------------------
  ## Determine "moist days", i.e. where soil moisture is above threshold.
  ## Get respective indices.
  ##------------------------------------------------
  ## If multiple layer's soil moisture data is available, do subset w.r.t. soil layer with highest value
  # df <- df %>% mutate( maxsoilm = apply( dplyr::select( df, one_of(settings$varnams_soilm) ), 1, FUN = max, na.rm = TRUE ) )
  # idxs_moist <- which( df$maxsoilm > soilm_threshold )
  
  ## simple
  idxs_moist <- which( df[[settings$varnams_soilm]] > soilm_threshold )
  
  ## identify and record moist days for this threshold
  vec_moist <- rep( NA, nrow(df) )
  vec_moist[idxs_moist]  <- TRUE
  vec_moist[-idxs_moist] <- FALSE
  
  ## If no weights are specified, use 1 for all data points
  if (is.na(weights)) weights <- rep( 1.0, nrow(df) )
  
  ##------------------------------------------------
  ## Loop and aggregate over repetitions
  ##------------------------------------------------
  out <- purrr::map(
    as.list(seq(settings$nrep)),
    ~train_predict_fvar_byrep( 
      ., 
      df,
      idxs_moist,
      settings, 
      weights = weights, 
      plot = FALSE,
      verbose = verbose 
    )
  )
  
  names(out) <- paste0("rep", seq(settings$nrep))
  
  df_all <- out %>% 
    purrr::map_dfr("df_all") %>% 
    dplyr::group_by(idx) %>%
    dplyr::summarize(
      nn_pot = mean(nn_pot, na.rm = TRUE),
      nn_act = mean(nn_act, na.rm = TRUE),
      fvar   = mean(nn_fxx, na.rm = TRUE)
    ) %>%
    dplyr::mutate(moist = vec_moist) %>%
    dplyr::mutate(fvar = remove_outliers_fXX(fvar, coef=3.0 )) %>% 
    
    ## re-append row ID (is first column)
    left_join(
      out %>% 
        purrr::map_dfr("df_all") %>% 
        dplyr::filter(irep == 1) %>% 
        dplyr::select(1, idx),
      by = "idx"
    ) %>% 
    
    ## complement with target variable and threshold variable (soil moisture)
    mutate(obs = df[[ settings$target ]],
           soilm = df[[ settings$varnams_soilm ]])
  
  df_cv <- out %>% 
    purrr::map_dfr("df_cv") %>% 
    dplyr::group_by(idx) %>%
    dplyr::summarize(
      nn_pot = mean(pred_pot, na.rm = TRUE),
      nn_act = mean(pred_act, na.rm = TRUE),
      obs    = mean(obs)
    ) %>%
    dplyr::mutate(moist = vec_moist)
  
  ## return only models from first repetitions
  return( list(nn_act = out$rep1$nn_act, nn_pot = out$rep1$nn_moist, df_all = df_all, df_cv = df_cv) )
}

train_predict_fvar_byrep <- function( irep, df, idxs_moist, settings, weights=NA, plot=FALSE, verbose ){
  
  ## Do training only if sufficient data is available
  if ( length(idxs_moist)>30 && (nrow(df) - length(idxs_moist))>30 ){
    
    if (verbose) rlang::inform(paste("NN repetition", irep))
    
    ##------------------------------------------------
    ## Train model on moist days' data, not using soil moisture as predictor ("_moist")
    ##------------------------------------------------
    ## Remove all variables starting with "soilm_" from predictors
    # settings$varnams_soilm <- df %>% dplyr::select( starts_with("soilm") ) %>% names()
    settings$predictors_without_soilm <- settings$predictors[ !(settings$predictors %in% settings$varnams_soilm) ]
    
    out_nn_moist <- predict_nn(
      data       = df[ idxs_moist, ],
      weights    = weights[ idxs_moist ],
      predictors = settings$predictors_without_soilm,
      nam_target = settings$target,
      do_predict = TRUE,
      package    = settings$package,
      lifesign   = "full",
      seed       = irep,
      hidden     = settings$nnodes_pot
    )#这里先启用nn吧，不用keras看会怎么样
   print(out_nn_moist)

    ## xxx try keras-specific. To do: make this function call flexible
    # out_nn_moist <- predict_nn_keras( df[ idxs_moist, ], 
    #                                   nam_target          = settings$target, 
    #                                   predictors          = settings$predictors_without_soilm, 
    #                                   prop                = 0.75, # make this a setting
    #                                   num_epochs          = 100, 
    #                                   batch_size          = 128, 
    #                                   val_size            = 0.15,
    #                                   learning_rate       = 0.001,
    #                                   num_layers          = 3,
    #                                   num_units_per_layer = 8,  # settings$nnodes_pot,
    #                                   print_model_summary = FALSE
    #                                   )

    
    # ## Evaluate predictions of good days model
    # stats_nn_moist <- rsofun::analyse_modobs(
    #   out_nn_moist$vals,
    #   df[[settings$target]][idxs_moist],
    #   plot.title="NN pot, moist only",
    #   do.plot=plot
    # )
    
    ##------------------------------------------------
    ## Get potential values (fvar_pot) with moist-days model, predicting all days
    ##------------------------------------------------
    ## Note: model is passed on as argument 'nn', therefore not training is done. 
    out_nn_pot <- predict_nn( 
      data       = df, 
      weights    = weights,
      predictors = settings$predictors,
      nam_target = settings$target, 
      nn         = out_nn_moist$nn, 
      do_predict = TRUE, 
      package    = settings$package
    )#这里的potential是针对所有数据的
    print(out_nn_pot)
    
    # ## Evaluate predictions of moist days model
    # stats_nn_pot <- rsofun::analyse_modobs(
    #   out_nn_pot$vals,
    #   df[[settings$target]],
    #   plot.title="NN pot",
    #   do.plot=plot
    # )
    
    ##------------------------------------------------
    ## Train full model, including soil moisture as a predictor ("act")
    ##------------------------------------------------
    out_nn_act <- predict_nn( 
      data       = df, 
      weights    = weights,
      predictors = settings$predictors, 
      nam_target = settings$target, 
      do_predict = TRUE, 
      package    = settings$package,
      seed       = irep,
      hidden     = settings$nnedoes_act
    )#这里的act应该是针对所有数据的
    print(out_nn_act)
    
    # ## get statistics of mod vs. obs of all-days full model
    # stats_nn_act <- rsofun::analyse_modobs( 
    #   out_nn_act$vals, 
    #   df[[settings$target]],
    #   plot.title = "NN act", 
    #   do.plot    = plot
    # )
    
    ##------------------------------------------------
    ## Construct data frame from validation results
    ##------------------------------------------------
    df_cv_act <- out_nn_act$df_cv %>% 
      setNames(paste0(names(.), "_act")) %>% 
      left_join(df %>% 
                  dplyr::select(settings$rowid) %>% 
                  mutate(idx_act = 1:n()),
                by = "idx_act")#by这里用的是包含_act行的ID；setNames(object,name);paste0(...,collapse = NULL)可以把一系列东西无空格地
    #连起来形成一个新的；所以这句话的功能是给原数据从左边加了一列表头为df_cv_act的列
    
    df_cv_pot <- out_nn_moist$df_cv %>%
      setNames(paste0(names(.), "_pot")) %>% 
      left_join(df[ idxs_moist, ] %>% 
                  dplyr::select(settings$rowid) %>% 
                  mutate(idx_pot = 1:n()),
                by = "idx_pot")#同上
    
    df_cv <- df_cv_act %>% 
      left_join(df_cv_pot, by = settings$rowid) %>% 
      dplyr::select(settings$rowid, obs = obs_act, pred_act, pred_pot) %>% 
      mutate(idx = 1:n())
    
    df_cv$moist <- rep(FALSE, nrow(df_cv))
    df_cv$moist[idxs_moist] <- TRUE
    
    out_nn_act$df_cv <- NULL
    out_nn_moist$df_cv <- NULL
    
    ##------------------------------------------------
    ## Construct predictions data frame with all
    ##------------------------------------------------
    df_all <- tibble(
      rowid  = df[[ settings$rowid ]],
      nn_pot = as.vector(out_nn_pot$df_all$pred),
      nn_act = as.vector(out_nn_act$df_all$pred),
      nn_fxx = as.vector(out_nn_act$df_all$pred) / as.vector(out_nn_pot$df_all$pred),
      irep   = irep
    ) %>% 
      mutate(idx = 1:n())
    
    ## rename row ID
    df_all <- df_all %>% setNames(c(settings$rowid, names(.)[-1]))
    
  } else {
    
    rlang::warn("Too few data points with current soil moisture threshold")
    
    df_all <- tibble(
      rowid  = df[[ settings$rowid ]],
      nn_pot = NA,
      nn_act = NA,
      nn_fxx = NA,
      irep   = irep
    ) %>% 
      mutate(idx = 1:n())
    
    ## rename row ID
    df_all <- df_all %>% setNames(c(settings$rowid, names(.)[-1]))
    
  }
  
  return(list(df_all = df_all, df_cv = df_cv, nn_act = out_nn_act, nn_moist = out_nn_moist))	
  
}



