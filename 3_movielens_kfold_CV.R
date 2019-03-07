
library(reshape2)
library(ggplot2)
library(irlba)


# Data processing (movielens 100 K) ---------------------------------------

df <- read.table(file = "data/movielens100k.data", sep = "\t")[,1:3]
colnames(df) <- c("user_id", "movie_id", "rating")
df[df$rating %in% c(1,2,3),3] <- -1
df[df$rating %in% c(4,5),3] <- 1

n_util_uniques <- length(unique(df$user_id))

# We randomly sample n users among the 943 unique users

n_users <- 200
set.seed(seed)
sample <- sample(x = c(1:n_util_uniques), size = n_users)
df <- df[df$user_id %in% sample,]

# Composition of test/training sets (80/20)

pct_test <- 0.2
n_test <- pct_test*nrow(df)
set.seed(seed)
sample_test <- sample(rownames(df), n_test)
df_test <- df[rownames(df) %in% sample_test,]

# In the training set, we set test set data to 0
df[rownames(df) %in% sample_test,3] <- 0

# Matrix to complete
M_data <- acast(df, user_id~movie_id, value.var="rating")
M_data[is.na(M_data)] <- 0 # Unobserved ratings are set to 0


# Functions declaration ---------------------------------------------------




loss_fun <- function(M_obs, B) {
  n_obs <- length(M_obs[M_obs != 0])
  M <- M_obs[M_obs != 0]
  B <- B[M_obs != 0]
  loss_mat <- log(1+exp(-M*B))
  return(mean(loss_mat))
}
# returns the value of the loss function g



grad_g <- function(M_obs, B) {
  n_obs <- length(M_obs[M_obs != 0])
  grad <- matrix(1/n_obs*(-M_obs)*exp(-M_obs*B)/(1+exp(-M_obs*B)), nrow = nrow(M_obs), ncol = ncol(M_obs))
  return(grad)
}
# returns the value of the gradient of the loss function g




soft_thresh <- function(B, lambda, t) {
  svd_list <- irlba(B)
  sv_vec <- svd_list$d
  sv_vec_thresh <- sapply(X = sv_vec, FUN = function(x) {ifelse(x >= lambda*t, x-lambda*t, 0)})
  sigma <- diag(sv_vec_thresh)
  soft_thresh_mat <- svd_list$u %*% sigma %*% t(svd_list$v)
  return(soft_thresh_mat)
}
# soft-thresholding of magnitude lambda*t to the B matrix



prox_grad <- function(M_obs, B, lambda, t) {
  grad_val <- grad_g(M_obs = M_obs, B = B)
  M_update <- soft_thresh(B = B - t*(grad_val), lambda = lambda, t = t)
  return(M_update)
}
# returns the updated matrix M after the proximal gradient step



penal_compute <- function(B, lambda) {
  svd_list <- irlba(B)
  nuc_norm_B <- sum(svd_list$d)
  penal <- lambda*nuc_norm_B
  return(penal)
}
# returns the value of the penalization function h (by the nuclear norm)



pgd <- function(M_obs, M_init, lambda, t, tol, maxiter) {
  
  M_prev <- M_init
  
  for (i in 1:maxiter) {
    
    M_upd <- prox_grad(M_obs = M_obs, B = M_prev, lambda, t)
    obj_val <- loss_fun(M_obs = M_obs, B = M_upd) + penal_compute(B = M_upd, lambda)
    
    evo <- norm(M_upd-M_prev, "F")/norm(M_prev, "F")
    if (is.nan(evo)) {
      stop("The value of lambda is too high, all the coefficients of M are equal to 0")
      
    } else if (evo < tol) break
    
    M_prev <- M_upd
  }
  return(M_upd)
}
# proximal gradient descent



cross_val <- function(M_data, lambda_min = 0, lambda_max, nb_lambda = 10, K = 5) {
  
  df <- setNames(melt(M_data), c('i', 'j', 'value'))
  df_obs <- df[df[,3] != 0,]
  
  # Random sampling of observed data
  set.seed(seed)
  df_obs <- df_obs[sample(nrow(df_obs)),]
  
  # Creates K identical folds
  folds <- cut(seq(1,nrow(df_obs)),breaks=K,labels=FALSE)
  
  # Build the training matrix and the test dataset for each fold
  for (k in 1:K) {
    
    testIndexes <- which(folds==k,arr.ind=TRUE)
    assign(paste0("df_test", k), df_obs[testIndexes, ])
    df_train <- df_obs
    df_train[testIndexes, 3] <- 0
    assign(paste0("df_train", k), df_train)

    M_train <- acast(df_train, i~j, value.var="value")
    M_train[is.na(M_train)] <- 0
    assign(paste0("M_train", k), M_train)
  }
  
  # Initialization (null) matrix
  
  m1_train <- nrow(M_train1)
  m2_train <- ncol(M_train1)
  M0 <- matrix(rep(0, m1_train*m2_train), nrow = m1_train, ncol = m2_train)
  
  # Cross-validation
  
  # Computes the prediction error on each fold for each lambda value
  range <- seq(from = lambda_max, to = lambda_min, length.out = nb_lambda)
  
  # Initializes vector of mean error rates on every bloc for each lambda
  mean_err <- rep(0, length(range)) 
  # Initializes list containing error rates on each bloc for each lambda
  list_err <- list()
  
  for (l in 1:length(range)) {
    
    lambda <- range[l]
    # Initializes vector of error rates for each bloc for the l-th lambda
    err_blocs <- rep(0,K)
    
    for (k in 1:K) {
      
      # Matrix completion on all training set except k-th bloc
      M_train <- get(paste0("M_train", k))
      M_pred <- pgd(M_obs = M_train, M_init = M0, lambda = lambda, t = 1, tol = 10^-2, maxiter = 500)
      rownames(M_pred) <- rownames(M_train)
      colnames(M_pred) <- colnames(M_train)
      
      M_estim <- sign(M_pred)
      
      # Add predicted ratings to test set
      df_test <- get(paste0("df_test", k))
      col_test <- colnames(df_test)
      pred_df <- setNames(melt(M_estim), c('i', 'j', 'pred'))
      compar_df <- merge(df_test,pred_df,by=c('i', 'j')) 
      
      # Comptes prediction error on fold k
      compar_df$pred_err <- ifelse(compar_df$value != compar_df$pred, 1, 0)
      err <- mean(compar_df$pred_err)
      err_blocs[k] <- err
    }
    mean_err[l] <- mean(err_blocs)
    list_err[[l]] <- err_blocs
    print(paste("It?ration", l, "sur", length(range)))
  }
  
  # Results of cross-validation
  
  num_min <- which(mean_err == min(mean_err))
  lambda_opti <- range[num_min]
  tx_pred_opti <- 1 - min(mean_err)
  conclu <- paste("The optimal value of lambda is", lambda_opti,
                  "giving a prediction rate of", tx_pred_opti)
  
  # Plotting results
  
  cv_plot_df <- data.frame("lambda" = range, "taux_erreur_moyen" = mean_err)
  
  # ggplot theme
  white_theme <- theme_bw() +
    theme(plot.title =element_text(size = rel(2.5)),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank())
  
  # cross-validation plot
  plot_cv <- ggplot(data = cv_plot_df, aes(x = lambda, y = taux_erreur_moyen)) + geom_point() + 
    geom_line() + white_theme 
  
  return(list(conclu, plot_cv, list_err))
  
}




# Cross-validation --------------------------------------------------------


# If lambda is too high, all the singular values of the matrix to complete
# are set to 0 => no convergence.
# By trial and error, we find a max value for lambda depending of n_users :
# 50 users => lambda_max = 0.003
# 200 users => lambda_max = 0.0009
# 500 users => lambda_max = 0.0005
# 943 users => lambda_max = 0.00035

cv_list <- cross_val(M_data = M_data, lambda_min = 0, lambda_max = 0.00035, 
                     nb_lambda = 10, K = 5)


# Results

cv_list[[1]] # Optimal lambda value and max prediction rate
cv_list[[2]] # Plot of mean error rate as a function of lambda
cv_list[[3]] # List of error rates on all blocs for each lambda (for verification)

