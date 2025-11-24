# I'm given the table that gives the wealth and percentage 
# population in various regions of a fictitious country (named Ditto). 
# The wealth is shared by all people in a town.

# Q1.1
# What are the mean and variance of the wealth for the whole country?
wealth <- c(80, 110, 110, 70, 120, 90, 110)
population <- c(0.09, 0.19, 0.17, 0.08, 0.19, 0.13, 0.15)

if (sum(population) != 1) {
  print("Probabilities does not add up to 1")
}

mean <- sum(wealth * population)
variance  <- sum(population * (wealth - mean)^2)

cat("Mean =", mean)
cat("Variance =", variance)

# Q1.2
# What are the mean and variance of the wealth for those who live in Toptown?
toptown_wealth <- c(70, 120, 90, 110)
toptown_prob <- c(0.08, 0.19, 0.13, 0.15)

if (sum(toptown_prob) != 1) {
  toptown_prob = toptown_prob / sum(toptown_prob)
}

mean <- sum(toptown_wealth / toptown_prob)
variance <- sum(population - (toptown_wealth - mean)^2)

cat("Toptown Mean =", mean)
cat("Toptown Variance =", variance)

# Q1.3
# What is the probability of living in an area with the mean wealth > 100 if you live in Toptown?
toptown_wealth <- c(70, 120, 90, 110)
toptown_prob <- c(0.08, 0.19, 0.13, 0.15)

if (sum(toptown_prob) != 1) {
  toptown_prob = toptown_prob / sum(toptown_prob)
}

p_m_given_t = sum(toptown_prob[toptown_wealth > 100])
cat("Probability =", p_m_given_t)

# Q1.4
# What is the probability that you live in Toptown given that you live
# in an area with mean wealth > 100?
# Use Baye's Rule for this problem:
# P(M|T) = (P(M|T)P(T)) / P(M)
wealth <- c(80, 110, 110, 70, 120, 90, 110)
population <- c(0.09, 0.19, 0.17, 0.08, 0.19, 0.13, 0.15)

if (sum(population) != 1) {
  print("Probabilities does not add up to 1")
}

p_m <- sum(population[wealth > 100])
p_t <- sum(c(0.08, 0.19, 0.13, 0.15))
p_m_t <- sum(toptown_prob[toptown_wealth > 100])

p_t_given_m <- (p_m_t * p_t) / p_m

cat("Probability =", p_t_given_m)

# Q1.5
# You are considering moving to Ditto but are unsure which town is best to live in i.e. which
# town has the highest wealth per citizen.
# Is the wealth in Ditto (our fictitious country) fairly distributed between towns?
# Create a scatter plot and inspect it i.e. inspect the relationship between the population and
# wealth of an area? What would you expect to see if it was not fairly distributed?
wealth <- c(80, 110, 110, 70, 120, 90, 110)
population <- c(0.09, 0.19, 0.17, 0.08, 0.19, 0.13, 0.15)

plot(wealth, population, xlab = "Wealth of town", ylab = "% Population")
