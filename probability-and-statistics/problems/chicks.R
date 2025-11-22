
attach(chickwts)
data <- chickwts

# Q1.A
boxplot(
  weight ~ feed,
  xlab = "Feed type",
  ylab = "Weight (grams)",
  main = "A Comparison of chick weight by feed type"
)

# Q1.B
# feeding chicks with sunflower, results in a higher mean weight
tapply(weight, feed, mean)
tapply(weight, feed, var)

# Q1.C
less_than_159 <- feed[weight < 159]
feed_linseed_or_sunflower <- less_than_159[less_than_159=="linseed" | less_than_159=="sunflower"]
print(feed_linseed_or_sunflower)
proportion <- length(feed_linseed_or_sunflower) / length(weight[weight < 159])
print(proportion)

# Q2
# It is known that 20% of products on a production
# line are defective. Products are inspected until the
# first defective product is encountered.
# Let X = number of inspections to obtain first defective
# P(defective) = 0.2, Geometric Distribution, x=5
# q ^ (x - 1) * p, where q = 1 - .2 = .8
p <- .2
fdni <- 0

for (i in c(1:n_iterations)) {
  j <- 1
  while (T) {
    if (runif(1) <= .2) {
      fdni[i] <- j
      break
    }
    j <- j + 1
  }
}
print(length(fdni[fdni == 5]) / n_iterations)
