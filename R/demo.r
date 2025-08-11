
library(ggplot2)
library(ggthemes)

set.seed(123)
df <- data.frame(
  x = rnorm(100),
  y = 2 * rnorm(100) + 3
)

p <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "#2C3E50", alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "#E74C3C", linewidth = 1.2) +
  ggtitle("高级散点图与回归线") +
  theme_economist() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title = element_text(size = 14)
  )
print(p)
