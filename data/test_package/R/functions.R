#' Add two numbers
#'
#' This function takes two numeric inputs and returns their sum.
#'
#' @param x A numeric value
#' @param y A numeric value
#' @return The sum of x and y
#' @examples
#' add_numbers(2, 3)
#' add_numbers(10, -5)
#' @export
add_numbers <- function(x, y) {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop("Both inputs must be numeric")
  }
  return(x + y)
}

#' Create a simple plot
#'
#' Generate a basic scatter plot using ggplot2.
#'
#' @param data A data frame with x and y columns
#' @param title Plot title
#' @return A ggplot object
#' @import ggplot2
#' @export
simple_plot <- function(data, title = "Simple Plot") {
  ggplot(data, aes(x = x, y = y)) +
    geom_point() +
    labs(title = title) +
    theme_minimal()
}
