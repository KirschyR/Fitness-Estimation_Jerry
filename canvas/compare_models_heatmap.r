# 比较两个模型输出的热图可视化
# 用于比较 VectorizedEvolutionModel 和 AgeStructuredPopulation 的输出差异

library(tidyverse)

# 设置工作目录
setwd("c:/Users/JerryR/Fitness-Estimation-master")

# 读取两个CSV文件
df_vectorized <- read.csv("simulation_results_(new_population_state).csv")
df_agestructured <- read.csv("simulation_results_(test_mosquito_population).csv")

# 选择需要比较的频率列（排除 generation, total_females, total_males）
freq_cols <- c(
  "female_DD", "female_DW", "female_DR", "female_WW", "female_WR", "female_RR",
  "male_DD", "male_DW", "male_DR", "male_WW", "male_WR", "male_RR",
  "allele_D", "allele_W", "allele_R"
)

# 提取频率数据
vec_freq <- df_vectorized[, freq_cols]
age_freq <- df_agestructured[, freq_cols]

# 计算差值矩阵（绝对差）
diff_matrix <- abs(vec_freq - age_freq)

# 将差值矩阵转换为长格式
diff_long <- diff_matrix %>%
  mutate(generation = df_vectorized$generation) %>%
  pivot_longer(
    cols = all_of(freq_cols),
    names_to = "variable",
    values_to = "difference"
  )

# 设置变量顺序
diff_long$variable <- factor(diff_long$variable, levels = freq_cols)

# 添加分组信息
diff_long <- diff_long %>%
  mutate(
    group = case_when(
      str_starts(variable, "female_") ~ "Female Genotypes",
      str_starts(variable, "male_") ~ "Male Genotypes",
      str_starts(variable, "allele_") ~ "Allele Frequencies"
    )
  )

# 分组热图
p_facet <- ggplot(diff_long, aes(x = factor(generation), y = variable, fill = difference)) +
  geom_tile(color = "grey80", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", difference)), size = 2.2, color = "black") +
  scale_fill_gradient2(
    name = "Absolute\nDifference",
    low = "#FFFFFF",
    mid = "#B3E5FC",
    high = "#0288D1",
    midpoint = max(diff_long$difference) / 2,
    limits = c(0, max(diff_long$difference))
  ) +
  facet_wrap(~ group, scales = "free_y", ncol = 1) +
  labs(
    title = "Model Comparison: Grouped by Variable Type",
    subtitle = "|VectorizedEvolutionModel - AgeStructuredPopulation|",
    x = "Generation",
    y = ""
  ) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    strip.text = element_text(size = 11, face = "bold"),
    strip.background = element_rect(fill = "#F5F5F5", color = NA),
    legend.position = "right",
    legend.background = element_rect(fill = "white", color = NA)
  )

ggsave("canvas/model_comparison_heatmap_grouped.png", p_facet, width = 12, height = 10, dpi = 150, bg = "white")
cat("分组热图已保存到: canvas/model_comparison_heatmap_grouped.png\n")

# ============================================================================
# 输出汇总统计
# ============================================================================

cat("\n==================== 差异汇总统计 ====================\n")

# 按变量汇总
summary_by_var <- diff_long %>%
  group_by(variable) %>%
  summarise(
    mean_diff = mean(difference),
    max_diff = max(difference),
    sum_diff = sum(difference),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_diff))

cat("\n按变量的平均差异（降序）：\n")
print(summary_by_var, n = 20)

# 按世代汇总
summary_by_gen <- diff_long %>%
  group_by(generation) %>%
  summarise(
    mean_diff = mean(difference),
    max_diff = max(difference),
    sum_diff = sum(difference),
    .groups = "drop"
  )

cat("\n按世代的平均差异：\n")
print(summary_by_gen, n = 20)

# 总体统计
cat("\n总体统计：\n")
cat(sprintf("  总体平均差异: %.6f\n", mean(diff_long$difference)))
cat(sprintf("  总体最大差异: %.6f\n", max(diff_long$difference)))
cat(sprintf("  差异最大的变量: %s (Gen %d, diff = %.6f)\n",
    diff_long$variable[which.max(diff_long$difference)],
    diff_long$generation[which.max(diff_long$difference)],
    max(diff_long$difference)))

cat("\n热图已生成完毕！\n")
cat(sprintf("  差异最大的变量: %s (Gen %d, diff = %.6f)\n",
    diff_long$variable[which.max(diff_long$difference)],
    diff_long$generation[which.max(diff_long$difference)],
    max(diff_long$difference)))

cat("\n所有热图已生成完毕！\n")
