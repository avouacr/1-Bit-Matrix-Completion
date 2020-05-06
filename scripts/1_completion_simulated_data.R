
library(reshape2)
library(ggplot2)




# Data processing ---------------------------------------------------------


# Simulation of a random binary matrix

L = matrix(rnorm(200),nc=2)
R = matrix(rnorm(200),nc=2)

m = L%*%t(R)
m = sign(m)

rm(L,R)

n <- nrow(m)*ncol(m)

# Building test set : 80% of m entries are randomly removed

df <- setNames(melt(m), c('i', 'j', 'value'))

n_test <- 0.8*n
set.seed(23)
sample_test <- sample(rownames(df), n_test)
df_test <- df[rownames(df) %in% sample_test,]

df[rownames(df) %in% sample_test,3] <- 0

# Creation of the matrice we are seeking to complete
M_data <- acast(df, i~j, value.var="value") 




# Algorithm ---------------------------------------------------------------




# Input parameters --------------------------------------------------------

n_obs <- length(df[df$value != 0,3])
m1 <- nrow(M_data)
m2 <- ncol(M_data)

# Initialization (null) matrix
M0 <- matrix(rep(0, m1*m2), nrow = m1, ncol = m2)


# Functions declaration ---------------------------------------------------


loss_fun <- function(M_data, B) {
  a <- length(M_data[M_data != 0])
  loss_val <- 0
  for (i in 1:length(M_data)) {
    if (M_data[i] != 0) {
      Y_i <- M_data[i]
      loss_val <- loss_val + log(1+exp((-Y_i)*B[i]))
    }
  }
  loss <- loss_val / a
  return(loss)
}
# returns the value of the loss function g


grad_g <- function(M_data, B) {
  grad <- matrix(data = rep(0, m1*m2), nrow = m1, ncol = m2)
  for (i in 1:length(M_data)) {
    if (M_data[i] != 0) {
      Y_i <- M_data[i]
      grad[i] <- grad[i] + 1/n_obs*(-Y_i)*exp(-Y_i*B[i])/(1+exp(-Y_i*B[i])) 
    }
  }
  return(grad)
}
# returns the value of the gradient of the loss function g


soft_thresh <- function(B, lambda, t) {
  svd_list <- svd(B)
  sv_vec <- svd_list$d
  sv_vec_thresh <- sapply(X = sv_vec, FUN = function(x) {ifelse(x >= lambda*t, x-lambda*t, 0)})
  sigma <- diag(sv_vec_thresh)
  soft_thresh_mat <- svd_list$u %*% sigma %*% t(svd_list$v)
  return(soft_thresh_mat)
}
# soft-thresholding of magnitude lambda*t to the B matrix


prox_grad <- function(M, lambda, t) {
  grad_val <- grad_g(M_data = M_data, B = M)
  M_update <- soft_thresh(B = M - t*(grad_val), lambda = lambda, t = t)
  return(M_update)
}
# returns the updated matrix M after the proximal gradient step


penal_compute <- function(B, lambda) {
  svd_list <- svd(B)
  nuc_norm_B <- sum(svd_list$d)
  penal <- lambda*nuc_norm_B
  return(penal)
}
# returns the value of the penalization function h (by the nuclear norm)



pgd <- function(M_data, M_init, lambda, t, tol, maxiter) {
  
  M_prev <- M_init
  
  for (i in 1:maxiter) {
    
    M_upd <- prox_grad(M = M_prev, lambda, t)
    obj_val <- loss_fun(M_data = M_data, B = M_upd) + penal_compute(B = M_upd, lambda)
    
    evo <- norm(M_upd-M_prev, "F")/norm(M_prev, "F")
    if (is.nan(evo)) {
      stop("The value of lambda is too high, all the coefficients of M are equal to 0")
      
    } else if (evo < tol) break
    
    M_prev <- M_upd
  }
  return(M_upd)
}
# proximal gradient descent


taux_pred <- function(M, df_test) {
  
  M_estim <- sign(M)
  
  # predicted ratings are added to the test dataset
  pred_df <- setNames(melt(M_estim), c('i', 'j', 'pred'))
  compar_df <- merge(df_test,pred_df,by=c("i","j")) 
  
  # computation of the prediction rate
  compar_df$pred_correcte <- ifelse(compar_df$value == compar_df$pred, 1, 0)
  (pred_rate <- mean(compar_df$pred_correcte))
}
# computes the prediction rate of a matrix on the test set


cross_val <- function(lambda_min = 0, lambda_max, nb_points = 10) {
  
  pred_vec <- c()
  range <- seq(from = lambda_min, to = lambda_max, length.out = nb_points)
  
  for (lambda in range) {
    
    # matrix completion for each value of lambda
    M <- pgd(M_data = M_data, M_init = M0, lambda = lambda, t = 1, tol = 10^-2, maxiter = 500)
    
    taux <- taux_pred(M = M, df_test = df_test)
    
    pred_vec <- c(pred_vec, taux)
  }
  num_max <- which(pred_vec == max(pred_vec))
  conclu <- paste("The optimal value of lambda is", range[num_max],
 "giving a prediction rate of", max(pred_vec))
  return(list(conclu, pred_vec))
}
# performs cross validation for a given range of lambda values
# returns the lambda value which maximizes the prediction rate and the max value


cv_plot <- function(lambda_min = 0, lambda_max, nb_points = 10) {
  
  range <- seq(from = lambda_min, to = lambda_max, length.out = nb_points)
  
  # Cross-validation
  cv_list <- cross_val(lambda_min = lambda_min, lambda_max = lambda_max, nb_points = nb_points)
  pred_vec <- cv_list[[2]]
  err_vec <- sapply(pred_vec, function(x) {1-x})
  cv_df <- data.frame("lambda" = range, "taux_erreur" = err_vec)
  
  # Ggplot theme
  white_theme <- theme_bw() +
    theme(plot.title =element_text(size = rel(2.5)),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank())
  
  plot_cv <- ggplot(data = cv_df, aes(x = lambda, y = taux_erreur)) + geom_point() + geom_line() + white_theme 
  return(plot_cv)
} 
# plots error rates for a given range of lambda values





# Proximal gradient descent -----------------------------------------------

M_test <- pgd(M_data = M_data, M_init = M0, lambda = 0, t = 1, 
              tol = 10^-2, maxiter = 500)



# Testing quality of the completion ---------------------------------------

taux_pred(M = M_test, df_test = df_test)



# Cross validation --------------------------------------------------------

# By trial and error, we find that for any lambda > 0.0032, the predicted
# matrix is set to zero. So we are restraining grid search to values
# of lambda below this threshold.

cv_list <- cross_val(lambda_min = 0, lambda_max = 0.0032, nb_points = 10)
cv_list[[1]]
cv_list[[2]]

# Plot cross-validation

a <- proc.time()
cv_plot(lambda_min = 0, lambda_max = 0.0032, nb_points = 10)
proc.time()-a
  









