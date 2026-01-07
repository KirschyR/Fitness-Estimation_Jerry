#!/usr/bin/env Rscript
library(rgl)

# 请把工作目录设为项目根（含 outputs2 文件夹）
# setwd("C:/Users/JerryR/Fitness-Estimation-master") 如需可取消注释并修改路径

outdir <- "outputs2"
week_dirs <- list.dirs(outdir, full.names = TRUE, recursive = FALSE)
week_dirs <- sort(week_dirs)
if (length(week_dirs) == 0) stop("未找到 outputs2 下的 week_* 目录")

# 可配置：哪些 (week, genotype) 对应的雌/雄柱子要并排（贴在一起）
# 格式：list(c("week_010", "WT|Drive"), c("week_005", "Drive|Drive"))
adjacency_list <- list()

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

  # 第一列是 age；其余列是基因型名称（列顺序应一致于所有周）
  if (is.null(geno_names)) {
    geno_names <- colnames(fdf)[-1]
  }

  # 按基因型汇总（对所有年龄求和）：得到每个基因型在该周的总个体数
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

zmax <- max(c(Fmat, Mmat), na.rm = TRUE)
if (zmax == 0) zmax <- 1

# ============ 布局参数 ============
inner_offset <- 0.3   # 雌雄柱子相互间的半偏移
cell_spacing_x <- 2   # 周之间的中心间隔
cell_spacing_y <- 2   # 基因型之间的中心间隔

# 绘图参数 - 使用基于间距计算的坐标轴范围
open3d()
bg3d("white")
plot3d(x = NA, y = NA, z = NA,
       xlim = c(0.5, nweeks * cell_spacing_x + 0.5),  # 修正：基于间距计算
       ylim = c(0.5, length(geno_names) * cell_spacing_y + 0.5),  # 修正：基于间距计算
       zlim = c(0, zmax * 1.1),
       xlab = "Week", ylab = "Genotype", zlab = "Count",
       box = FALSE, axes = FALSE)
axes3d(edges = c("x--", "y--", "z--"))

# 刻度与标签 - 修正：基于间距计算标签位置
xticks <- seq_len(nweeks)
xlabels <- week_names
yticks <- seq_along(geno_names)
ylabels <- geno_names

# 周标签：x位置 = i * cell_spacing_x
for (i in xticks) {
  x_pos <- i * cell_spacing_x  # 修正：基于间距计算
  text3d(x_pos, 0.3, 0, texts = xlabels[i], cex = 0.7)
}

# 基因型标签：y位置 = j * cell_spacing_y
for (j in yticks) {
  y_pos <- j * cell_spacing_y  # 修正：基于间距计算
  text3d(0.3, y_pos, 0, texts = ylabels[j], cex = 0.7)
}

mtext3d("Week", edge = "x+", line = 2)
mtext3d("Genotype", edge = "y+", line = 2)
mtext3d("Count", edge = "z+", line = 2)

# 更稳健的绘制立方体柱函数（从 z=0 向上绘制，忽略非正高度）
draw_bar <- function(x, y, height, wx = 0.3, wy = 0.3, col = "red") {
  if (is.na(height)) return()
  h <- as.numeric(height)
  if (is.nan(h) || h <= 0) return()
  h <- max(h, 1e-8)
  cube <- rgl::cube3d()
  cube <- rgl::scale3d(cube, wx, wy, h)
  cube <- rgl::translate3d(cube, x, y, h)
  rgl::shade3d(cube, color = col, alpha = 0.95)
}

# 在 (x,y) 网格上绘制，每个格子中心为 (iw*cell_spacing_x, ig*cell_spacing_y)
for (iw in seq_len(nweeks)) {
  for (ig in seq_along(geno_names)) {
    x_center <- iw * cell_spacing_x
    y_center <- ig * cell_spacing_y

    # 左为 female，右为 male，确保两根柱子相贴
    fx <- x_center - inner_offset
    mx <- x_center + inner_offset

    fz <- Fmat[ig, iw]
    mz <- Mmat[ig, iw]
    draw_bar(fx, y_center, fz, wx = 0.35, wy = 0.35, col = "#FF69B4")
    draw_bar(mx, y_center, mz, wx = 0.35, wy = 0.35, col = "#87CEFA")
  }
}

# 添加地面参考平面 (z=0) - 已正确使用间距
rgl::quads3d(
  x = c(0, nweeks * cell_spacing_x + 1, nweeks * cell_spacing_x + 1, 0),
  y = c(0, 0, length(geno_names) * cell_spacing_y + 1, length(geno_names) * cell_spacing_y + 1),
  z = c(0, 0, 0, 0),
  color = "#F0F0F0", alpha = 0.6
)

# 图例
legend3d("topright", legend = c("Female", "Male"), pch = 15, col = c("#FF69B4", "#87CEFA"), cex = 1)

# 使用提示：若希望调整间距或柱宽，修改 inner_offset / cell_spacing_x / cell_spacing_y / wx/wy 参数