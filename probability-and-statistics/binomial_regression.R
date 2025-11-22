# The number of ready terminals
# (in 5) where the probability of a 
# terminal being ready is 95%.
# Calculating PMD for xє[1..5]
x <- 0:5
plot(
  x,
  dbinom(x, 5, .95),
  xlab = "Number of Ready Terminals",
  ylab = "P(X = x)",
  type = "h",
  main = "Ready Terminals, n = 5, p = .95"
)

# The probability that X or less
# terminals (in 5) will be ready
# where the probability of a
# terminal being ready is .95.
# Calculating CDF
x <- 0:5
plot(
  x,
  pbinom(x, 5, .95),
  xlab = "Number of Ready Terminals",
  ylab = "P(X = x)",
  ylim = c(0, 1),
  type = "S",
  main = "n = 5, p = .95"
)

# Suppose there are n frames per packet and 
# the probability that a frame gets through
# without an error is .999. What is the maximum
# size that a packet can be so that the probability
# that it contains no frame in error is at least .9 ?
n <- 1:200
plot(
  n,
  .999^n,
  type = "l",
  ylab = "P(Frames in error = 0)",
  xlab="Number of frames in a packet",
  ylim= c(.9, .999)
)
