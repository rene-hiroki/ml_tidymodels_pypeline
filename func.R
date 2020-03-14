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




# for classification

# roc curve and auc -------------------------------------------------------

roc_curve_and_auc <- function(res_models) {
  tmp_auc <-
    res_solo_models %>%
    map(roc_auc, !!target_var, !!pred_positive) %>%
    map2(.x = .,
         .y = names(res_solo_models),
         ~ mutate(.x, model = .y)) %>%
    bind_rows() %>% arrange(.metric, desc(.estimate))
  
  p <- res_solo_models %>%
    map2(names(.), ~ mutate(.x, model = .y)) %>%
    map(group_by, model) %>%
    map(roc_curve,
        !!target_var, !!pred_positive) %>%
    bind_rows() %>%
    autoplot(size = ) +
    labs(title = "ROC Curve")
  
  return(list(auc = tmp_auc, roc = p))
}


# threshold_f1score -------------------------------------------------------

threshold_f1score <-
  function(res_fit, lo=0.5, hi=0.95) {
    res <- tibble()
    for (thre in seq(lo, hi, 0.01)) {
      tmp <-
        res_fit %>%
        select(target_var, contains(paste0("pred_", positive))) %>%
        pivot_longer(-target_var,
                     names_to = "model",
                     values_to = "prob") %>%
        group_by(model) %>%
        mutate(truth = !!target_var,
               pred = ifelse(prob >= thre, "WS", "PS")) %>%
        mutate(pred = as.factor(pred)) %>%
        f_meas(truth, pred) %>%
        mutate(threshold = thre) %>%
        rename(f1_score = .estimate)
      res <- tmp %>% bind_rows(res)
    }
    
    adjust_thre <-
      res %>%
      group_by(model) %>%
      arrange(threshold) %>%
      distinct() %>%
      arrange(desc(f1_score)) %>%
      distinct(model, .keep_all = TRUE)
    
    p <- res %>%
      ggplot(aes(x = threshold, y = f1_score)) +
      geom_point() +
      labs(title = paste0("adjusted threshold for miximize f1_score is: ", adjust_thre$threshold)) +
      geom_vline(xintercept = adjust_thre$threshold) +
      geom_hline(yintercept = adjust_thre$f1_score) +
      scale_x_continuous(n.breaks = 10) +
      scale_y_continuous(n.breaks = 10)
    
    cat("\n res_f1, adjusted_threshold, and roc_plot are returned. \n")
    
    return(list(
      res_f1 = res,
      adjusted_threshold = adjust_thre,
      thre_f1 = p
    ))
  }


# threshold_f1score all ---------------------------------------------------------------

threshold_f1score_all <-
  function(res_fit, lo=0.05, hi=0.95) {
    res <- tibble()
    for (thre in seq(lo, hi, 0.01)) {
      tmp <-
        res_fit %>%
        select(target_var, contains(paste0("pred_", positive))) %>%
        pivot_longer(-target_var,
                     names_to = "model",
                     values_to = "prob") %>%
        group_by(model) %>%
        mutate(truth = !!target_var,
               pred = ifelse(prob >= thre, "WS", "PS")) %>%
        mutate(pred = as.factor(pred)) %>%
        f_meas(truth, pred) %>%
        mutate(threshold = thre) %>%
        rename(f1_score = .estimate)
      res <- tmp %>% bind_rows(res)
    }
    
    adjust_thre <-
      res %>%
      group_by(model) %>%
      arrange(threshold) %>%
      distinct() %>%
      arrange(desc(f1_score)) %>%
      distinct(model, .keep_all = TRUE)
    
    p <- res %>%
      ggplot(aes(x = threshold, y = f1_score, color = model)) +
      geom_point() +
      labs(title = paste0("all mothreshold vs f1_score")) +
      scale_x_continuous(n.breaks = 10) +
      scale_y_continuous(n.breaks = 10)
    
    cat("\n res_f1, adjusted_threshold, and roc_plot are returned. \n")
    
    return(list(
      res_f1 = res,
      adjusted_threshold = adjust_thre,
      roc_plot = p
    ))
  }


# predict by adjusted threshold -------------------------------------------

predict_by_adjusted_threshold <-
  function(res_models, adjusted_thre_models) {
    res <-
      map2(.x = res_models,
           .y = adjusted_thre_models,
           ~ mutate(
             .x,
             pred = ifelse(
               !!pred_positive > .y$adjusted_threshold$threshold,
               positive,
               negative
             ),
             pred = as.factor(pred)
           ))
    cat("\n add pred column to the data by using adjusted threshold\n")
    return(res)
  }



# confusion matrix all ----------------------------------------------------

confusion_matrix_info <-
  function(res_models) {
    confusion_matrix_all <-
      res_models %>%
      map(.x = .,
          ~ confusionMatrix(.x$pred, .x[[target_var]], positive = positive) %>% tidy)
    
    confusion_matrix_all <-
      confusion_matrix_all %>%
      bind_rows() %>%
      mutate(model = rep(names(confusion_matrix_all),
                         each = nrow(confusion_matrix_all[[1]]))) %>%
      arrange(term, desc(estimate))
    info <-
      res_models %>%
      map(.x = .,
          ~ confusionMatrix(.x$pred, .x[[target_var]], positive = positive))
    
    cm <- res_models %>% map(conf_mat,
                             truth = !!target_var,
                             estimate = pred)
    hm <- cm %>% map(autoplot, "heatmap") %>%
      map2(names(cm),wrapper_add_title)
    ms <- cm %>% map(autoplot, "mosaic") %>% 
      map2(names(cm),wrapper_add_title)
    
    
    return(
      list(
        info = info,
        confusion_matrix_all = confusion_matrix_all,
        heatmap = hm,
        mosaic = ms,
        cm = cm
      )
    )
  }


# wrapper add title gg ----------------------------------------------------

wrapper_add_title <- function(p, model_names){
  p + labs(title = model_names)
}
