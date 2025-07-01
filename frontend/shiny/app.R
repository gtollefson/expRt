# R Package Assistant - Shiny Frontend
# Interactive interface for querying the local LLM assistant

library(shiny)
library(shinydashboard)
library(DT)
library(httr)
library(jsonlite)
library(markdown)

# Configuration
API_BASE_URL <- Sys.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY <- Sys.getenv("API_KEY", "dev-key-123")

# Helper function to call API
call_api <- function(endpoint, method = "GET", data = NULL) {
  url <- paste0(API_BASE_URL, endpoint)
  
  headers <- add_headers(
    "Authorization" = paste("Bearer", API_KEY),
    "Content-Type" = "application/json"
  )
  
  if (method == "GET") {
    response <- GET(url, headers)
  } else if (method == "POST") {
    response <- POST(url, headers, body = toJSON(data, auto_unbox = TRUE))
  }
  
  if (status_code(response) == 200) {
    return(fromJSON(content(response, "text")))
  } else {
    return(list(error = paste("API Error:", status_code(response))))
  }
}

# UI
ui <- dashboardPage(
  dashboardHeader(title = "🚀 R Package Assistant"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Query Assistant", tabName = "query", icon = icon("search")),
      menuItem("Statistics", tabName = "stats", icon = icon("chart-bar")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        .query-box {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin-bottom: 20px;
        }
        .answer-box {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin-bottom: 20px;
          min-height: 200px;
        }
        .source-item {
          background: #f8f9fa;
          padding: 15px;
          margin: 10px 0;
          border-left: 4px solid #007bff;
          border-radius: 4px;
        }
      "))
    ),
    
    tabItems(
      # Query tab
      tabItem(tabName = "query",
        fluidRow(
          column(12,
            div(class = "query-box",
              h3("Ask about R packages and functions"),
              p("Enter your question about R programming, package usage, or function documentation."),
              
              textAreaInput("query_text", 
                label = NULL,
                placeholder = "Example: How do I create a scatter plot with ggplot2?",
                width = "100%",
                height = "80px"
              ),
              
              fluidRow(
                column(6,
                  selectInput("package_filter", "Filter by Package:", 
                    choices = c("All packages" = "", "ggplot2", "dplyr", "tidyr", "base"),
                    selected = ""
                  )
                ),
                column(6,
                  selectInput("file_type_filter", "Filter by File Type:",
                    choices = c("All types" = "", "R source (.R)" = ".R", 
                               "Help files (.Rd)" = ".Rd", "Vignettes (.Rmd)" = ".Rmd"),
                    selected = ""
                  )
                )
              ),
              
              fluidRow(
                column(6,
                  numericInput("max_results", "Max Results:", value = 5, min = 1, max = 20)
                ),
                column(6,
                  br(),
                  actionButton("submit_query", "Search", 
                    class = "btn-primary",
                    style = "margin-top: 5px;"
                  )
                )
              )
            )
          )
        ),
        
        # Results section
        conditionalPanel(
          condition = "output.show_results",
          fluidRow(
            column(12,
              div(class = "answer-box",
                h4("Answer"),
                div(id = "loading", 
                  style = "display: none;",
                  p("🔍 Searching through R documentation...")
                ),
                htmlOutput("answer_text")
              )
            )
          ),
          
          fluidRow(
            column(12,
              h4("Sources"),
              p("Relevant documentation chunks used to generate the answer:"),
              htmlOutput("sources_list")
            )
          ),
          
          fluidRow(
            column(12,
              h4("Query Details"),
              verbatimTextOutput("query_info")
            )
          )
        )
      ),
      
      # Statistics tab
      tabItem(tabName = "stats",
        fluidRow(
          column(6,
            box(
              title = "Vector Store Statistics", status = "primary", solidHeader = TRUE,
              width = NULL,
              tableOutput("vector_stats")
            )
          ),
          column(6,
            box(
              title = "Model Information", status = "info", solidHeader = TRUE,
              width = NULL,
              tableOutput("model_info")
            )
          )
        ),
        
        fluidRow(
          column(12,
            box(
              title = "Performance Metrics", status = "success", solidHeader = TRUE,
              width = NULL,
              tableOutput("performance_stats")
            )
          )
        )
      ),
      
      # About tab
      tabItem(tabName = "about",
        fluidRow(
          column(12,
            box(
              title = "About R Package Assistant", status = "primary", solidHeader = TRUE,
              width = NULL,
              HTML("
                <h4>🚀 Local LLM-Powered R Package Assistant</h4>
                <p>This application provides intelligent assistance for R package development and usage.</p>
                
                <h5>Features:</h5>
                <ul>
                  <li><strong>Version-aware documentation</strong> - Prevents outdated syntax issues</li>
                  <li><strong>Local processing</strong> - No data sent to external services</li>
                  <li><strong>Fast retrieval</strong> - FAISS-powered vector search</li>
                  <li><strong>Context-aware answers</strong> - Understands R ecosystem</li>
                </ul>
                
                <h5>Technology Stack:</h5>
                <ul>
                  <li><strong>Frontend:</strong> R Shiny</li>
                  <li><strong>Backend:</strong> FastAPI + Python</li>
                  <li><strong>Embeddings:</strong> Sentence Transformers</li>
                  <li><strong>Vector Store:</strong> FAISS</li>
                  <li><strong>LLM:</strong> Local models (Phi-3, Mistral)</li>
                </ul>
                
                <h5>Development Status:</h5>
                <p>This is a development version with mock responses. The full pipeline includes:</p>
                <ol>
                  <li>R package documentation chunking</li>
                  <li>Semantic embedding generation</li>
                  <li>Vector store creation and indexing</li>
                  <li>Local LLM inference</li>
                  <li>Context-aware response generation</li>
                </ol>
              ")
            )
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Reactive values
  values <- reactiveValues(
    query_result = NULL,
    show_results = FALSE
  )
  
  # Query submission
  observeEvent(input$submit_query, {
    if (nchar(trimws(input$query_text)) == 0) {
      showNotification("Please enter a question.", type = "warning")
      return()
    }
    
    # Show loading
    shinyjs::show("loading")
    values$show_results <- TRUE
    
    # Prepare API request
    query_data <- list(
      query = input$query_text,
      max_results = input$max_results,
      package_filter = if(input$package_filter == "") NULL else input$package_filter,
      file_type_filter = if(input$file_type_filter == "") NULL else input$file_type_filter
    )
    
    # Call API
    tryCatch({
      result <- call_api("/query", "POST", query_data)
      values$query_result <- result
      shinyjs::hide("loading")
      
      if (!is.null(result$error)) {
        showNotification(paste("Error:", result$error), type = "error")
      } else {
        showNotification("Query completed successfully!", type = "success")
      }
    }, error = function(e) {
      shinyjs::hide("loading")
      showNotification(paste("Connection error:", e$message), type = "error")
    })
  })
  
  # Show results condition
  output$show_results <- reactive({
    values$show_results
  })
  outputOptions(output, "show_results", suspendWhenHidden = FALSE)
  
  # Answer display
  output$answer_text <- renderUI({
    if (is.null(values$query_result) || !is.null(values$query_result$error)) {
      return(p("No results yet."))
    }
    
    answer <- values$query_result$answer
    # Convert markdown-style formatting to HTML
    answer_html <- gsub("\\n", "<br>", answer)
    answer_html <- gsub("\\*\\*(.*?)\\*\\*", "<strong>\\1</strong>", answer_html)
    
    HTML(answer_html)
  })
  
  # Sources display
  output$sources_list <- renderUI({
    if (is.null(values$query_result) || !is.null(values$query_result$error)) {
      return(p("No sources available."))
    }
    
    sources <- values$query_result$sources
    if (length(sources) == 0) {
      return(p("No sources found."))
    }
    
    source_items <- lapply(1:length(sources), function(i) {
      source <- sources[[i]]
      div(class = "source-item",
        h5(paste("Source", i, "- Score:", round(source$score, 3))),
        p(strong("File:"), source$source_file),
        if (!is.null(source$package_name)) p(strong("Package:"), source$package_name),
        if (!is.null(source$section_header)) p(strong("Section:"), source$section_header),
        p(strong("Preview:"), source$text_preview)
      )
    })
    
    do.call(tagList, source_items)
  })
  
  # Query info
  output$query_info <- renderText({
    if (is.null(values$query_result) || !is.null(values$query_result$error)) {
      return("No query information available.")
    }
    
    info <- values$query_result
    paste(
      paste("Query time:", round(info$query_time, 3), "seconds"),
      paste("Embedding model:", info$model_info$embedding_model),
      paste("LLM model:", info$model_info$llm_model),
      paste("Vector store:", info$model_info$vector_store),
      sep = "\n"
    )
  })
  
  # Statistics tab
  observe({
    # Load stats on tab change
    stats_result <- call_api("/stats")
    
    if (!is.null(stats_result$error)) {
      output$vector_stats <- renderTable(data.frame(
        Metric = "Error",
        Value = stats_result$error
      ))
    } else {
      # Vector store stats
      vs_stats <- stats_result$vector_store
      output$vector_stats <- renderTable(data.frame(
        Metric = c("Total Chunks", "Packages", "R Files", "Help Files", "Vignettes"),
        Value = c(
          vs_stats$total_chunks,
          length(vs_stats$packages),
          vs_stats$file_types$`.R` %||% 0,
          vs_stats$file_types$`.Rd` %||% 0,
          vs_stats$file_types$`.Rmd` %||% 0
        )
      ))
      
      # Model info
      model_info <- stats_result$models
      output$model_info <- renderTable(data.frame(
        Component = c("Embedding Model", "Embedding Dimension", "LLM Model"),
        Details = c(
          model_info$embedding_model,
          model_info$embedding_dim,
          model_info$llm_model
        )
      ))
      
      # Performance stats
      perf_stats <- stats_result$performance
      output$performance_stats <- renderTable(data.frame(
        Metric = c("Average Query Time", "Total Queries Processed"),
        Value = c(
          paste(perf_stats$avg_query_time, "seconds"),
          perf_stats$total_queries
        )
      ))
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server) 