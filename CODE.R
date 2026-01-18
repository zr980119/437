library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(corrplot)
library(gridExtra)
library(scales)
library(ggridges)
library(glmnet)
theme_set(theme_minimal(base_size = 11))
options(scipen = 999)
set.seed(123)

dir.create("figures", showWarnings = FALSE)

df_raw <- read_csv("billboard_24years_lyrics_spotify.csv", show_col_types = FALSE)

# DATASET OVERVIEW
print(nrow(df_raw))
print(ncol(df_raw))
print(min(df_raw$year, na.rm = TRUE))
print(max(df_raw$year, na.rm = TRUE))


# MISSING VALUES
missing_counts <- colSums(is.na(df_raw))
print(missing_counts[missing_counts > 0])


# AUDIO FEATURES AVAILABILITY
audio_features <- c("danceability", "energy", "valence", "acousticness",
                    "speechiness", "liveness", "loudness", "tempo")
complete_audio <- sum(complete.cases(df_raw[, audio_features]))
print(complete_audio)
print(round(complete_audio / nrow(df_raw) * 100, 2))


# DATA CLEANING AND PREPARATION

df_clean <- df_raw %>%
  filter(complete.cases(.[, audio_features])) %>%
  distinct(song, band_singer, year, .keep_all = TRUE) %>%
  mutate(
    # Binary target: Top 50 vs Lower 50
    top_50 = factor(
      ifelse(ranking <= 50, "Top50", "Lower50"),
      levels = c("Top50", "Lower50")
    ),
    
    # Categorical ranking groups
    rank_category = factor(
      case_when(
        ranking <= 10 ~ "Top 10",
        ranking <= 25 ~ "11-25",
        ranking <= 50 ~ "26-50",
        ranking <= 100 ~ "51-100"
      ),
      levels = c("Top 10", "11-25", "26-50", "51-100")
    ),
    
    # Time periods
    period = factor(
      case_when(
        year >= 2000 & year < 2005 ~ "2000-2004",
        year >= 2005 & year < 2010 ~ "2005-2009",
        year >= 2010 & year < 2015 ~ "2010-2014",
        year >= 2015 & year < 2020 ~ "2015-2019",
        year >= 2020 ~ "2020+"
      ),
      levels = c("2000-2004", "2005-2009", "2010-2014", 
                 "2015-2019", "2020+")
    ),
    
    # Normalized duration
    duration_min = duration_ms / 60000
  )

# CLEANED DATASET
print(nrow(df_clean))
print(n_distinct(df_clean$band_singer))

# TARGET VARIABLE DISTRIBUTION
print(table(df_clean$top_50))
print(round(prop.table(table(df_clean$top_50)) * 100, 2))

# RANKING CATEGORY DISTRIBUTION
print(table(df_clean$rank_category))

# TIME PERIOD DISTRIBUTION
print(table(df_clean$period))


# EXPLORATORY DATA ANALYSIS

# DESCRIPTIVE STATISTICS
summary_stats <- df_clean %>%
  select(all_of(audio_features), ranking, duration_min) %>%
  summary()
print(summary_stats)


# Distribution of Audio Features
feature_dist <- df_clean %>%
  select(all_of(audio_features)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  mutate(feature = str_to_title(feature))

png("figures/eda_01_feature_distributions.png", 
    width = 12, height = 10, units = "in", res = 300, bg = "white")
ggplot(feature_dist, aes(x = value)) +
  geom_histogram(bins = 30, fill = "#440154", alpha = 0.7, color = "white") +
  geom_density(aes(y = after_stat(count) * 0.03), 
               color = "#FDE725", size = 1.2) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  labs(
    title = "Distribution of Audio Features in Billboard Hot-100 Songs",
    subtitle = paste0("N = ", nrow(df_clean), " songs with complete Spotify features"),
    x = "Feature Value",
    y = "Count",
    caption = "Source: Billboard Hot-100 (2000-2021) with Spotify Audio Features"
  ) +
  theme(
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "gray40")
  )
dev.off()

# Top 50 vs Lower 50 Comparison
comparison_data <- df_clean %>%
  select(top_50, all_of(audio_features)) %>%
  pivot_longer(cols = -top_50, names_to = "feature", values_to = "value") %>%
  mutate(feature = str_to_title(feature))

png("figures/eda_02_top50_comparison.png", 
    width = 12, height = 10, units = "in", res = 300, bg = "white")
ggplot(comparison_data, aes(x = value, fill = top_50)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c("Top50" = "#440154", "Lower50" = "#FDE725"),
                    labels = c("Top 50", "Lower 50")) +
  labs(
    title = "Audio Feature Distributions: Top 50 vs Lower 50 Rankings",
    subtitle = "Comparing density distributions to identify distinguishing characteristics",
    x = "Feature Value",
    y = "Density",
    fill = "Ranking Group",
    caption = "Overlapping distributions suggest features alone may not strongly predict ranking"
  ) +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 11),
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank()
  )
dev.off()

# STATISTICAL TESTS (Top 50 vs Lower 50)
for(feature in audio_features) {
  test_result <- t.test(
    df_clean[[feature]][df_clean$top_50 == "Top50"],
    df_clean[[feature]][df_clean$top_50 == "Lower50"]
  )
  print(sprintf("%-15s: t = %6.3f, p = %.4f %s",
                str_to_title(feature),
                test_result$statistic,
                test_result$p.value,
                ifelse(test_result$p.value < 0.05, "***", "")))
}


# Correlation Matrix
cor_data <- df_clean %>%
  select(all_of(audio_features), ranking)

cor_matrix <- cor(cor_data, use = "complete.obs")

png("figures/eda_03_correlation_matrix.png", 
    width = 10, height = 10, units = "in", res = 300, bg = "white")
corrplot(cor_matrix,
         method = "circle",
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 1,
         addCoef.col = "black",
         number.cex = 0.75,
         col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200),
         title = "\n\nCorrelation Matrix: Audio Features and Chart Ranking",
         mar = c(0, 0, 3, 0))
dev.off()

# CORRELATION WITH RANKING
ranking_cors <- cor_matrix[, "ranking"]
print(sort(abs(ranking_cors), decreasing = TRUE))


# Temporal Evolution
temporal_data <- df_clean %>%
  filter(!is.na(period)) %>%
  group_by(period, top_50) %>%
  summarise(
    avg_danceability = mean(danceability),
    avg_energy = mean(energy),
    avg_valence = mean(valence),
    avg_acousticness = mean(acousticness),
    n = n(),
    .groups = 'drop'
  ) %>%
  pivot_longer(
    cols = starts_with("avg_"),
    names_to = "feature",
    values_to = "value"
  ) %>%
  mutate(feature = str_remove(feature, "avg_"),
         feature = str_to_title(feature))

png("figures/eda_04_temporal_evolution.png", 
    width = 14, height = 8, units = "in", res = 300, bg = "white")
ggplot(temporal_data, aes(x = period, y = value, 
                          color = top_50, group = top_50)) +
  geom_line(size = 1.2) +
  geom_point(size = 3, alpha = 0.7) +
  facet_wrap(~ feature, scales = "free_y", ncol = 4) +
  scale_color_manual(values = c("Top50" = "#440154", "Lower50" = "#FDE725"),
                     labels = c("Top 50", "Lower 50")) +
  labs(
    title = "Temporal Evolution of Audio Features (2000-2021)",
    subtitle = "Comparing trends between Top 50 and Lower 50 songs across time periods",
    x = "Time Period",
    y = "Average Feature Value",
    color = "Ranking Group",
    caption = "Limited data in 2020+ period"
  ) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 16)
  )
dev.off()

# Ranking Category Detailed Analysis
rank_analysis <- df_clean %>%
  group_by(rank_category) %>%
  summarise(
    across(all_of(audio_features), 
           list(mean = mean, sd = sd),
           .names = "{.col}_{.fn}"),
    n = n(),
    .groups = 'drop'
  )

rank_means <- rank_analysis %>%
  select(rank_category, n, ends_with("_mean")) %>%
  pivot_longer(cols = ends_with("_mean"),
               names_to = "feature",
               values_to = "mean") %>%
  mutate(feature = str_remove(feature, "_mean"),
         feature = str_to_title(feature))

png("figures/eda_05_ranking_categories.png", 
    width = 12, height = 8, units = "in", res = 300, bg = "white")
ggplot(rank_means, aes(x = rank_category, y = mean, 
                       group = 1)) +
  geom_line(color = "#440154", size = 1) +
  geom_point(color = "#440154", size = 3) +
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  labs(
    title = "Average Audio Features Across Ranking Categories",
    subtitle = "Exploring gradual changes from Top 10 to 51-100 rankings",
    x = "Ranking Category",
    y = "Average Value",
    caption = paste0("Sample sizes: Top 10 (n=", 
                     rank_analysis$n[1], "), 11-25 (n=",
                     rank_analysis$n[2], "), 26-50 (n=",
                     rank_analysis$n[3], "), 51-100 (n=",
                     rank_analysis$n[4], ")")
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 16)
  )
dev.off()

#  FEATURE ENGINEERING
df_model <- df_clean %>%
  mutate(
    # Interaction features
    energy_loud_ratio = energy / (abs(loudness) + 1),
    dance_energy = danceability * energy,
    valence_dance = valence * danceability,
    
    # Composite features
    positivity_score = (valence + danceability) / 2,
    acoustic_natural = acousticness * (1 - speechiness),
    
    # Normalized tempo
    tempo_z = scale(tempo)[,1],
    
    # Duration category
    duration_category = cut(
      duration_min,
      breaks = c(0, 3, 3.5, 4, 10),
      labels = c("Short", "Medium", "Long", "Very Long")
    )
  )

# FEATURE CREATION SUMMARY
print(sum(!names(df_model) %in% names(df_clean)))
print(ncol(df_model))


# PREDICTIVE MODELING SETUP

# Select features for modeling
predictor_features <- c(
  audio_features,
  "energy_loud_ratio", "dance_energy", "valence_dance",
  "positivity_score", "acoustic_natural", "tempo_z"
)

# MODELING SETUP
print(length(predictor_features))
print(predictor_features)


# Train-test split (70-30)
train_index <- createDataPartition(df_model$top_50, p = 0.7, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

# TRAINING AND TEST SET SIZES
print(nrow(train_data))
print(nrow(test_data))

# Training and test set distribution
print(table(train_data$top_50))
print(table(test_data$top_50))


#  MODEL 1 - RANDOM FOREST
rf_model <- randomForest(
  x = train_data[, predictor_features],
  y = train_data$top_50,
  ntree = 500,
  mtry = 4,
  importance = TRUE,
  nodesize = 10,
  maxnodes = 50
)

# Predictions
rf_train_pred <- predict(rf_model, train_data[, predictor_features])
rf_test_pred <- predict(rf_model, test_data[, predictor_features])

# Performance metrics
rf_train_acc <- mean(rf_train_pred == train_data$top_50)
rf_test_acc <- mean(rf_test_pred == test_data$top_50)

# Random forest performance
print(round(rf_train_acc, 4))
print(round(rf_test_acc, 4))
print(round(rf_model$err.rate[500, "OOB"], 4))


# Confusion matrix
rf_cm <- confusionMatrix(rf_test_pred, test_data$top_50)
# Confusion matrix
print(rf_cm$table)
# Detailed metrics
print(rf_cm$byClass)


# Feature Importance
importance_df <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(
    feature = str_replace_all(feature, "_", " "),
    feature = str_to_title(feature),
    feature = reorder(feature, MeanDecreaseGini)
  )

png("figures/model_01_rf_importance.png", 
    width = 10, height = 8, units = "in", res = 300, bg = "white")
ggplot(importance_df, aes(x = feature, y = MeanDecreaseGini)) +
  geom_col(fill = "#440154", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    subtitle = "Features ranked by Mean Decrease in Gini impurity",
    x = NULL,
    y = "Mean Decrease Gini",
    caption = paste0("OOB Error Rate: ", 
                     round(rf_model$err.rate[500, "OOB"] * 100, 2), "%")
  ) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    axis.text.y = element_text(size = 10)
  )
dev.off()

# Top 10 most important features
print(head(importance_df[, c("feature", "MeanDecreaseGini")], 10))


# RF Confusion Matrix
cm_data <- as.data.frame(rf_cm$table) %>%
  mutate(
    Prediction = factor(Prediction, levels = c("Top50", "Lower50")),
    Reference = factor(Reference, levels = c("Top50", "Lower50"))
  )

png("figures/model_02_rf_confusion.png", 
    width = 9, height = 8, units = "in", res = 300, bg = "white")
ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", size = 2) +
  geom_text(aes(label = Freq), size = 16, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#FDE725", high = "#440154") +
  labs(
    title = "Random Forest Confusion Matrix",
    subtitle = paste0("Test Accuracy: ", round(rf_test_acc * 100, 2), 
                      "% | Sensitivity: ", round(rf_cm$byClass["Sensitivity"] * 100, 2),
                      "% | Specificity: ", round(rf_cm$byClass["Specificity"] * 100, 2), "%"),
    x = "Actual Ranking",
    y = "Predicted Ranking",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    axis.text = element_text(size = 12, face = "bold"),
    legend.position = "right"
  )
dev.off()

# MODEL 2 - LOGISTIC REGRESSION
# Prepare binary outcome
train_data_glm <- train_data %>%
  mutate(top_50_binary = as.numeric(top_50 == "Top50"))
test_data_glm <- test_data %>%
  mutate(top_50_binary = as.numeric(top_50 == "Top50"))

# Fit model
glm_formula <- as.formula(paste("top_50_binary ~", 
                                paste(predictor_features, collapse = " + ")))
glm_model <- glm(glm_formula, data = train_data_glm, family = binomial)

print(summary(glm_model))

# Predictions
glm_train_prob <- predict(glm_model, train_data_glm, type = "response")
glm_test_prob <- predict(glm_model, test_data_glm, type = "response")

glm_train_pred <- factor(
  ifelse(glm_train_prob > 0.5, "Top50", "Lower50"),
  levels = c("Top50", "Lower50")
)
glm_test_pred <- factor(
  ifelse(glm_test_prob > 0.5, "Top50", "Lower50"),
  levels = c("Top50", "Lower50")
)

# Performance metrics
glm_train_acc <- mean(glm_train_pred == train_data$top_50)
glm_test_acc <- mean(glm_test_pred == test_data$top_50)

# Logistic regression performance
print(round(glm_train_acc, 4))
print(round(glm_test_acc, 4))


# ROC analysis
roc_obj <- roc(test_data_glm$top_50_binary, glm_test_prob)
auc_value <- auc(roc_obj)
# AUC
print(round(auc_value, 4))


# ROC Curve
png("figures/model_03_glm_roc.png", 
    width = 10, height = 9, units = "in", res = 300, bg = "white")
plot(roc_obj,
     col = "#440154",
     lwd = 3,
     main = "ROC Curve: Logistic Regression Model",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = FALSE,
     print.thres = FALSE,
     grid = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = alpha("#440154", 0.2))
abline(a = 0, b = 1, lty = 2, col = "gray60", lwd = 2)
text(0.6, 0.2, 
     paste0("AUC = ", round(auc_value, 3)),
     cex = 1.5, font = 2, col = "#440154")
text(0.6, 0.1,
     paste0("Test Accuracy = ", round(glm_test_acc * 100, 2), "%"),
     cex = 1.2, col = "gray30")
dev.off()

# Coefficient analysis
coef_summary <- summary(glm_model)$coefficients
coef_df <- data.frame(
  feature = rownames(coef_summary)[-1],
  coefficient = coef_summary[-1, "Estimate"],
  p_value = coef_summary[-1, "Pr(>|z|)"]
) %>%
  mutate(
    significant = ifelse(p_value < 0.05, "Yes", "No"),
    feature = str_replace_all(feature, "_", " "),
    feature = str_to_title(feature)
  ) %>%
  arrange(desc(abs(coefficient)))

# Significant coefficients (p < 0.05)
print(coef_df %>% filter(significant == "Yes") %>%
        select(feature, coefficient, p_value))

# Coefficient Plot
png("figures/model_04_glm_coefficients.png", 
    width = 10, height = 8, units = "in", res = 300, bg = "white")
ggplot(coef_df, aes(x = reorder(feature, coefficient), 
                    y = coefficient,
                    fill = significant)) +
  geom_col(alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  coord_flip() +
  scale_fill_manual(values = c("Yes" = "#440154", "No" = "#FDE725"),
                    labels = c("p < 0.05", "p >= 0.05")) +
  labs(
    title = "Logistic Regression Coefficients",
    subtitle = "Effect size and statistical significance of predictors",
    x = NULL,
    y = "Coefficient Estimate",
    fill = "Significant",
    caption = "Positive values increase odds of being Top 50"
  ) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    axis.text.y = element_text(size = 10),
    legend.position = "top"
  )
dev.off()

# MODEL COMPARISON
comparison_results <- data.frame(
  Model = c("Random Forest", "Logistic Regression"),
  Train_Accuracy = c(rf_train_acc, glm_train_acc),
  Test_Accuracy = c(rf_test_acc, glm_test_acc),
  OOB_AUC = c(NA, auc_value)
)

print(comparison_results)

comparison_long <- comparison_results %>%
  select(-OOB_AUC) %>%
  pivot_longer(cols = c(Train_Accuracy, Test_Accuracy),
               names_to = "Metric",
               values_to = "Accuracy") %>%
  mutate(Metric = str_replace(Metric, "_", " "))

# Performance Comparison
png("figures/model_05_comparison.png", 
    width = 11, height = 7, units = "in", res = 300, bg = "white")
ggplot(comparison_long, aes(x = Model, y = Accuracy, fill = Metric)) +
  geom_col(position = "dodge", width = 0.6, alpha = 0.9) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 2), "%")),
            position = position_dodge(width = 0.6),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("Train Accuracy" = "#440154", 
                               "Test Accuracy" = "#FDE725")) +
  scale_y_continuous(labels = percent_format(),
                     limits = c(0, 1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Predictive Model Performance Comparison",
    subtitle = paste0("Random Forest AUC: N/A | Logistic Regression AUC: ", 
                      round(auc_value, 3)),
    x = NULL,
    y = "Classification Accuracy",
    fill = "Dataset",
    caption = "Both models show modest predictive power, suggesting chart success depends on factors beyond audio features"
  ) +
  theme(
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    axis.text.x = element_text(size = 12, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 11)
  )
dev.off()

# ERROR ANALYSIS
# Combine predictions
error_analysis <- test_data %>%
  mutate(
    rf_prediction = rf_test_pred,
    glm_prediction = glm_test_pred,
    glm_probability = glm_test_prob,
    rf_correct = (rf_prediction == top_50),
    glm_correct = (glm_prediction == top_50),
    both_correct = rf_correct & glm_correct,
    both_wrong = !rf_correct & !glm_correct
  )

# Error analysis summary
print(sum(error_analysis$both_correct))
print(sum(error_analysis$both_wrong))
print(sum(error_analysis$rf_correct & !error_analysis$glm_correct))
print(sum(error_analysis$glm_correct & !error_analysis$rf_correct))

# Analyze misclassified songs
misclassified <- error_analysis %>%
  filter(both_wrong) %>%
  select(song, band_singer, ranking, top_50, 
         all_of(audio_features), glm_probability)

# Characteristics of misclassified songs
print(nrow(misclassified))


# Average features (misclassified vs all)
for(feature in audio_features) {
  mis_avg <- mean(misclassified[[feature]], na.rm = TRUE)
  all_avg <- mean(test_data[[feature]], na.rm = TRUE)
  print(sprintf("%-15s: Misclassified = %.3f, All = %.3f, Diff = %.3f",
                str_to_title(feature), mis_avg, all_avg, mis_avg - all_avg))
}



# Prediction Confidence
pred_conf_data <- test_data_glm %>%
  mutate(
    predicted = glm_test_pred,
    probability = glm_test_prob,
    confidence = abs(probability - 0.5),
    correct = (predicted == top_50)
  )

png("figures/analysis_01_prediction_confidence.png", 
    width = 12, height = 7, units = "in", res = 300, bg = "white")
ggplot(pred_conf_data, aes(x = probability, fill = correct)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  geom_vline(xintercept = 0.5, linetype = "dashed", 
             color = "red", size = 1.2) +
  scale_fill_manual(values = c("TRUE" = "#440154", "FALSE" = "#FDE725"),
                    labels = c("Correct", "Incorrect")) +
  scale_x_continuous(labels = percent_format()) +
  labs(
    title = "Logistic Regression Prediction Confidence Distribution",
    subtitle = "Distribution of predicted probabilities colored by correctness",
    x = "Predicted Probability of Top 50",
    y = "Count",
    fill = "Prediction",
    caption = "Threshold at 0.5 (dashed line)"
  ) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 16)
  )
dev.off()

# Feature Importance Comparison
top_features_rf <- head(importance_df$feature, 8)
top_features_glm <- head(coef_df$feature, 8)

feature_importance_comp <- data.frame(
  Feature = c(top_features_rf, top_features_glm),
  Model = c(rep("Random Forest", 8), rep("Logistic Regression", 8)),
  Rank = c(1:8, 1:8)
)

png("figures/analysis_02_feature_ranking.png", 
    width = 12, height = 8, units = "in", res = 300, bg = "white")
ggplot(feature_importance_comp, 
       aes(x = Rank, y = reorder(Feature, -Rank), 
           color = Model, shape = Model)) +
  geom_point(size = 5, alpha = 0.8) +
  geom_line(aes(group = Feature), color = "gray70", size = 0.8) +
  scale_color_manual(values = c("Random Forest" = "#440154", 
                                "Logistic Regression" = "#FDE725")) +
  scale_x_continuous(breaks = 1:8) +
  labs(
    title = "Feature Importance Rankings: Model Comparison",
    subtitle = "Top 8 features from each model",
    x = "Importance Rank",
    y = NULL,
    color = "Model",
    shape = "Model",
    caption = "Connected features show same feature ranked by both models"
  ) +
  theme(
    panel.grid.major.y = element_line(color = "gray90"),
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "top",
    axis.text.y = element_text(size = 10)
  )
dev.off()



# Dataset statistics
print(nrow(df_raw))
print(nrow(df_clean))
print(n_distinct(df_clean$band_singer))
print(min(df_clean$year))
print(max(df_clean$year))
print(nrow(train_data))
print(nrow(test_data))


# Class distribution
print(table(df_clean$top_50))
print(round(prop.table(table(df_clean$top_50))[1] * 100, 2))

# Model performance
print(round(rf_test_acc * 100, 2))
print(round(rf_model$err.rate[500, "OOB"] * 100, 2))
print(round(glm_test_acc * 100, 2))
print(round(auc_value, 4))

# Top 5 important features (random forest)
print(head(importance_df[, c("feature", "MeanDecreaseGini")], 5))


