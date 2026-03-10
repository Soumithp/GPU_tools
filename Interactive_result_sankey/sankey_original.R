setwd("C:/Users/sowmi/Desktop/4th Semester/Computational biologist_sankeyplot")

library(networkD3)
library(dplyr)
library(manipulateWidget)
library(htmlwidgets)
library(htmltools)


df <- read.table("example.txt", header = TRUE)

filtered_data <- df[df$p < 0.01, ]

filtered_data <- filtered_data[(filtered_data$paired_tstat < 0 & filtered_data$risk_association == "High_risk") |
                                 (filtered_data$paired_tstat > 0 & filtered_data$risk_association == "Low_risk"), ]

sankey_data <- data.frame(source = character(),
                          target = character(),
                          value = numeric(),
                          risk = character(),
                          link_group = character(),
                          stringsAsFactors = FALSE)

# Iterate over the rows of the filtered_data dataframe
for (i in 1:nrow(filtered_data)) {
  # Get the values for the current row
  gene <- filtered_data$gene[i]
  drug <- filtered_data$drug[i]
  value <- abs(filtered_data$paired_tstat[i])
  risk <- filtered_data$risk_association[i]
  
  link_group <- ifelse( drug %in% filtered_data$drug, drug)
  
  
  # Add the row to the sankey_data dataframe based on the risk_association value
  if (risk == "High_risk") {
    row <- c(gene, drug, value, risk, link_group)
    
  } else if (risk == "Low_risk") {
    row <- c(drug, gene, value, risk, link_group)
  }
  
  # Append the row to the sankey_data dataframe
  sankey_data <- rbind(sankey_data, row)
}

colnames(sankey_data) <- c("source", "target", "value", "risk", "link_group")

# Create a new column for node groups
unique_sources <- unique(sankey_data$source)
unique_targets <- unique(sankey_data$target)
unique_nodes <- unique(c(unique_sources, unique_targets))
nodes <- data.frame(name = unique_nodes,
                    NodeGroup = ifelse(unique_nodes %in% filtered_data$drug, unique_nodes, "gene"))


sankey_data$IDsource <- match(sankey_data$source, nodes$name) - 1
sankey_data$IDtarget <- match(sankey_data$target, nodes$name) - 1

 
  
my_color <- 'd3.scaleOrdinal() .domain([ "AM095" , "Cenicriviroc" , "EGCG" ,
                 "Erlotinib", "Galunisertib" , "Metformin" ,
                 "MG132"]) .range(["red", "blue", "green", "Grey", "cyan", "pink","orange" ])'


sankeyPlot <- sankeyNetwork(Links = sankey_data, Nodes = nodes, Source = "IDsource", Target = "IDtarget",
                            Value = "value", NodeID = "name", NodeGroup = "NodeGroup",
                            LinkGroup = "link_group",
                            colourScale = my_color, fontSize = 9,
                            nodeWidth = 10, nodePadding = 5, margin = list("left" = 460, "right" = 460, "top"= 10, "bottom"=10),
                            sinksRight = F)





sankeyPlot


#write.table(filtered_data, file="C:/Users/sowmi/Desktop/4th Semester/Computational biologist_sankeyplot", sep="\t" , quote = FALSE, row.names= FALSE)

file_path <- "C:/Users/sowmi/Desktop/4th Semester/Computational biologist_sankeyplot/sankey_drug_genepairs.txt"

write.table(filtered_data, file = file_path, sep = "\t", quote = FALSE, row.names = FALSE)

close(file_path)




