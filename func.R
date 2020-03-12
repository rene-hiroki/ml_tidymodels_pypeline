# load library ------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(naniar)     # vis_miss
library(janitor)    # rename colnames to under score
library(furrr)      # parallel
library(rlang)      # expression
library(earth)      # modeling mars
library(xgboost)    # modeling xgboost
library(lightgbm)   # modeling lgbm
library(Matrix)     # modeling lgbm
library(tictoc)     # time check
library(patchwork)  # plot



# feature importance ------------------------------------------------------

my_importance_plot <- function(fit_boost, res_boost) {
  # feature importance
  ifelse(
    any(class(fit_boost) == "lgb.Booster"),
    vi <- fit_boost %>%
      lgb.importance() %>%
      xgb.ggplot.importance() +
      theme(legend.position = "none") +
      labs(title = "lgb: Feature importance"),
    vi <- fit_boost$fit %>%
      xgb.importance(model = .) %>%
      xgb.ggplot.importance() +
      theme(legend.position = "none") +
      labs(title = "xgb: Feature importance")
  )
  
  hi <- res_boost %>%
    pivot_longer(
      cols = c(target_var, fit),
      names_to = "pred_truth",
      values_to = "value"
    ) %>%
    ggplot(aes(value)) +
    geom_histogram() +
    facet_wrap(vars(pred_truth))
  
  sc <- res_boost %>%
    ggplot(aes(!!target_var, fit)) +
    geom_point()
  
  a <- (vi | (hi / sc))
  return(a)
}


# lgb data set ------------------------------------------------------------


create_lgb_dataset <- function(target_var, rec_preped, test) {
  # set target label
  target <-
    rec_preped %>%
    juice() %>%
    pull(target_var)
  # create dgc matrics
  lgb_train_mat <-
    rec_preped %>%
    juice() %>%
    select(-target_var) %>%
    as.matrix() %>%
    Matrix(sparse = TRUE)
  
  # lgb_data
  lgb_data <- lgb.Dataset(data = lgb_train_mat, label = target)
  
  lgb_test_mat <-
    test %>%
    bake(rec_preped, new_data = .) %>%
    select(-target_var) %>%
    as.matrix() %>%
    Matrix(sparse = TRUE)
  
  cat("target, lgb_train_mat, lgb_data, and lgb_test_mat are created \n")
  
  return(
    list(
      target = target,
      lgb_train_mat = lgb_train_mat,
      lgb_data = lgb_data,
      lgb_test_mat = lgb_test_mat
    )
  )
  
}


# fit cv no tuning --------------------------------------------------------

fit_cv_no_tuning <- function(model_list_no_tuning) {
  tic.clear()
  tic("fitting time is")
  fits <-
    future_map(
      .x = model_list_no_tuning,
      .f = ~ fit_resamples(
        rec,
        .x,
        data_cv,
        control = control_resamples(save_pred = TRUE)
      )
    )
  t <- toc()
  time <- t$toc - t$tic
  tic.clear()
  
  
  return(list(
    fits = fits,
    time = paste0("fitting time is: ", time)
  ))
  
  cat("\nfits is returned\n")
  cat("\nfitting time is: ", time, "\n")
  
}



# fit cv tuning -----------------------------------------------------------

fit_cv_tuning <- function(model_list_tuning, grid_list) {
  tic.clear()
  tic("fitting time is")
  fits <-
    future_map2(
      .x = model_list_tuning,
      .y = grid_list,
      .f = ~ tune_grid(
        rec,
        .x,
        resamples = data_cv,
        grid = .y,
        control = control_resamples(save_pred = TRUE)
      )
    )
  t <- toc()
  time <- t$toc - t$tic
  tic.clear()
  
  return(list(
    fits = fits,
    time = paste0("fitting time is: ", time)
  ))
  
  cat("\nfits is returned\n")
  cat("\nfitting time is: ", time, "\n")
}


# plot test to confirm ----------------------------------------------------

my_test_confirm_plot <- function(pred_model) {
  
  hi <- baked_test %>%
    pivot_longer(
      cols = c(target_var, pred_model),
      names_to = "pred_truth",
      values_to = "value"
    ) %>%
    ggplot(aes(value)) +
    geom_histogram() +
    facet_grid(vars(pred_truth))
  
  sc <-  baked_test %>%
    ggplot(aes(!!target_var, .data[[pred_model]])) +
    geom_point()
  
  a <- ((hi | sc))
  return(a)
}

