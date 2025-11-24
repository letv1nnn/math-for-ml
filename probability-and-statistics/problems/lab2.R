# Question 1
# In this question we are going to do random simulations of a binomial distribution and a
# hypergeometric distribution.
#
# Imagine that a box contains 15 white balls and 5 black balls. Suppose that we draw a sample
# of 5 balls at random from the box. Let the random variable X be the number of black balls in
# the sample. Clearly the value of X could vary from 0 to 5.
#
# If we draw the sample with replacement, then X should obey a binomial distribution. If we
# draw the sample without replacement, then X should obey a hypergeometric distribution.
#
# You can use the R function, sample, to simulate drawing 5 balls at random from the box. You
# can read the documentation on this function using help

balls <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

# Q1.a
b_n = sample(balls, 5, replace = T)

# Q1.b
n_iters = 10000
black_balls = rep(0, n_iters)

for (i in c(1:n_iters)) {
  b_n <- sample(balls, 5, replace = T)
  b_c <- sum(b_n == 0)
  black_balls[i] <- b_c
}
cat("With Replacement - Distribution:\n")
table(black_balls) / n_iters

# Q1.c
p = sum(balls == 0) / length(balls)  # Probability of black ball
cat("Q1.c - Binomial Distribution (with replacement):\n")
for (k in 0:5) {
  binomial_prob = dbinom(k, size = 5, prob = p)  # Fixed: size = 5, not 20
  cat("P( X =", k, ") =", round(binomial_prob, 4), "\n")
}
cat("\n")

# Q1.d
m_black = sum(balls == 0)
n_white = sum(balls == 1)
k_draw = 5
for (k in c(0:k_draw)) {
  hypergeom_prob = dhyper(k, m_black, n_white, k_draw)
  cat("P( X =", k, ") =", round(hypergeom_prob, 4), "\n")
}
