
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

n_users <- 943
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

m1 <- nrow(M_data)
m2 <- ncol(M_data)

# Initialization (null) matrix
M0 <- matrix(rep(0, m1*m2), nrow = m1, ncol = m2)




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



cross_val <- function(M_data, lambda_min = 0, lambda_max, nb_lambda = 10, tol = 10^-3) {
  
  range <- seq(from = lambda_max, to = lambda_min, length.out = nb_lambda)
  err_vec <- c()
  
  for (i in 1:length(range)) {
    
    # Proximal Gradient Descent
    M_pred <- pgd(M_obs = M_data, M_init = M0, lambda = range[i], t = 1, tol = tol, maxiter = 500)
    rownames(M_pred) <- rownames(M_data)
    colnames(M_pred) <- colnames(M_data)
    
    # Transformation to a binary matrix
    M_estim <- sign(M_pred)
    
    # Predicted ratings are added to the test set
    pred_df <- setNames(melt(M_estim), c('user_id', 'movie_id', 'pred'))
    compar_df <- merge(df_test,pred_df,by=c("user_id","movie_id"))
    
    # Computation of the prediction rate
    compar_df$pred_err <- ifelse(compar_df$rating != compar_df$pred, 1, 0)
    err_rate <- mean(compar_df$pred_err)
    
    err_vec <- c(err_vec, err_rate)
    
    #print(paste("It?ration", i, "sur", length(range)))
    
  }
  
  # First result : optimal lambda value and prediction rate on the test set
  
  num_min <- which(err_vec == min(err_vec))
  lambda_opti <- range[num_min]
  tx_pred_opti <- 1 - min(err_vec)
  conclu <- paste("The optimal value of lambda is", lambda_opti,
                  "giving a prediction rate of", tx_pred_opti)
  
  # Second result : plot of the error rate as a function of lambda
  
  cv_results_df <- data.frame("lambda" = range, "taux_erreur" = err_vec)
  
  # ggplot theme
  white_theme <- theme_bw() +
    theme(plot.title =element_text(size = rel(2.5)),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank())
  
  # cross-validation plot
  plot_cv <- ggplot(data = cv_results_df, aes(x = lambda, y = taux_erreur)) + geom_point() + 
    geom_line() + white_theme 
  
  return(list(conclu, plot_cv))
  
}





# Cross validation --------------------------------------------------------

# If lambda is too high, all the singular values of the matrix to complete
# are set to 0 => no convergence.
# By trial and error, we find a max value for lambda depending of n_users :
# 50 users => lambda_max = 0.0023
# 200 users => lambda_max = 0.0009
# 500 users => lambda_max = 0.0005
# 943 users => lambda_max = 0.0004

cv_list <- cross_val(M_data = M_data, lambda_min = 0, lambda_max = 0.0004,
                     nb_lambda = 10, tol = 10^-2)


# Results

cv_list[[1]] # Optimal lambda value and max prediction rate
cv_list[[2]] # Plot of error rate as a function of lambda


