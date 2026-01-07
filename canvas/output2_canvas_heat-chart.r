#!/usr/bin/env Rscript
library(ggplot2)
library(reshape2)
library(pheatmap)
library(RColorBrewer)

# 请把工作目录设为项目根（含 outputs2 文件夹）
setwd("C:/Users/JerryR/Fitness-Estimation-master")

outdir <- "outputs2"
week_dirs <- list.dirs(outdir, full.names = TRUE, recursive = FALSE)
week_dirs <- sort(week_dirs)
if (length(week_dirs) == 0) stop("未找到 outputs2 下的 week_* 目录")

female_mat_list <- list()
male_mat_list <- list()
geno_names <- NULL
week_names <- character()

for (wdir in week_dirs) {
  fpath <- file.path(wdir, "individual_count", "individual_count_female.csv")
  mpath <- file.path(wdir, "individual_count", "individual_count_male.csv")
  if (!file.exists(fpath) || !file.exists(mpath)) {
    warning("缺少文件，跳过: ", wdir); next
  }
  fdf <- read.csv(fpath, stringsAsFactors = FALSE, check.names = FALSE)
  mdf <- read.csv(mpath, stringsAsFactors = FALSE, check.names = FALSE)

  # 第一列是 age；其余列是基因型名称
  if (is.null(geno_names)) {
    geno_names <- colnames(fdf)[-1]
  }

  # 按基因型汇总（对所有年龄求和）
  fcounts <- colSums(fdf[ , -1, drop = FALSE], na.rm = TRUE)
  mcounts <- colSums(mdf[ , -1, drop = FALSE], na.rm = TRUE)

  # 确保列顺序与 geno_names 匹配
  fcounts <- fcounts[geno_names]
  mcounts <- mcounts[geno_names]

  female_mat_list[[length(female_mat_list) + 1]] <- as.numeric(fcounts)
  male_mat_list[[length(male_mat_list) + 1]] <- as.numeric(mcounts)
  week_names <- c(week_names, basename(wdir))
}

nweeks <- length(female_mat_list)
if (nweeks == 0) stop("没有可用周数据")

# 转换为矩阵：行 = genotype, 列 = week
Fmat <- do.call(cbind, female_mat_list)
Mmat <- do.call(cbind, male_mat_list)
rownames(Fmat) <- geno_names
rownames(Mmat) <- geno_names
colnames(Fmat) <- week_names
colnames(Mmat) <- week_names

# ==================== 方法1：使用ggplot2创建热图 ====================

# 创建数据框用于ggplot2
create_plot_data <- function(mat, sex) {
  df <- as.data.frame(mat)
  df$Genotype <- rownames(df)
  df_melted <- melt(df, id.vars = "Genotype", variable.name = "Week", value.name = "Count")
  df_melted$Sex <- sex
  return(df_melted)
}

female_data <- create_plot_data(Fmat, "Female")
male_data <- create_plot_data(Mmat, "Male")
plot_data <- rbind(female_data, male_data)

# 调整因子顺序，使图形更美观
plot_data$Genotype <- factor(plot_data$Genotype, levels = rev(geno_names))
plot_data$Week <- factor(plot_data$Week, levels = week_names)

# ==================== 对数变换处理 ====================
plot_data$LogCount <- log10(plot_data$Count + 1)

# ==================== 对数变换后的热图 ====================
p2 <- ggplot(plot_data, aes(x = Week, y = Genotype, fill = LogCount)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_gradientn(
    colors = brewer.pal(9, "YlOrRd"),
    name = expression(paste(log[10], "(Count+1)")),
    na.value = "grey90",
    breaks = pretty(range(plot_data$LogCount, na.rm = TRUE), n = 5),
    labels = function(x) round(10^x - 1, 1)
  ) +
  # 上下分布，移除自由缩放参数以兼容coord_fixed
  facet_grid(Sex ~ .) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", color = "white"),  # 整个图形背景为白色
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 10),
    panel.grid = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    legend.position = "right",
    strip.background = element_rect(fill = "lightgray", color = "gray")
  ) +
  labs(
    title = "Individual Counts by Genotype and Week (Log10 Scale)",
    x = "Week",
    y = "Genotype"
  ) +
  coord_fixed(ratio = 0.8)  # 保持方格为正方形

print(p2)
out_png <- file.path(script_dir, "heatmap_log_vertical.png")
ggsave(out_png, p2, width = 12, height = 8, dpi = 300)
cat("Saved heatmap to:", out_png, "\n")

# ==================== 新增：标准化后的热图（按行标准化） ====================
# 对每个基因型的数据进行标准化（Z-score或min-max标准化）
# 这样可以更清楚地看到每个基因型内部的变化模式

# # 首先计算每个基因型的标准化值
# standardize_data <- function(data, method = "zscore") {
#   if (method == "zscore") {
#     # Z-score标准化：(x - mean)/sd
#     data %>%
#       group_by(Genotype, Sex) %>%
#       mutate(
#         Mean = mean(Count, na.rm = TRUE),
#         SD = sd(Count, na.rm = TRUE),
#         StdCount = ifelse(SD > 0, (Count - Mean) / SD, 0)
#       ) %>%
#       ungroup() %>%
#       select(-Mean, -SD)
#   } else {
#     # Min-Max标准化：(x - min)/(max - min)
#     data %>%
#       group_by(Genotype, Sex) %>%
#       mutate(
#         Min = min(Count, na.rm = TRUE),
#         Max = max(Count, na.rm = TRUE),
#         StdCount = ifelse(Max > Min, (Count - Min) / (Max - Min), 0.5)
#       ) %>%
#       ungroup() %>%
#       select(-Min, -Max)
#   }
# }

# plot_data_std <- standardize_data(plot_data, method = "zscore")

# p3 <- ggplot(plot_data_std, aes(x = Week, y = Genotype, fill = StdCount)) +
#   geom_tile(color = "white", size = 0.5) +
#   scale_fill_gradient2(
#     low = "blue",
#     mid = "white",
#     high = "red",
#     midpoint = 0,
#     name = "Z-score",
#     na.value = "grey90"
#   ) +
#   facet_grid(Sex ~ ., scales = "free_y", space = "free_y") +
#   theme_minimal(base_size = 12) +
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     axis.text.y = element_text(size = 10),
#     panel.grid = element_blank(),
#     strip.text = element_text(size = 12, face = "bold"),
#     legend.position = "right",
#     strip.background = element_rect(fill = "lightgray", color = "gray")
#   ) +
#   labs(
#     title = "Individual Counts by Genotype and Week (Z-score Standardized)",
#     x = "Week",
#     y = "Genotype",
#     subtitle = "Standardized within each genotype to highlight patterns"
#   ) +
#   coord_fixed(ratio = 0.8)

# print(p3)

# 保存对数尺度热图
# ggsave(file.path(script_dir, "heatmap_log_vertical.png"), p2, width = 10, height = 8, dpi = 300)

# # 保存标准化热图
# ggsave(file.path(script_dir, "heatmap_standardized_vertical.png"), p3, width = 10, height = 8, dpi = 300)
# cat("标准化热图已保存为:", file.path(script_dir, "heatmap_standardized_vertical.png"), "\n")