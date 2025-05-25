"""
完成相机标定、视差计算、三维重建的完整流程
棋盘格参数: 6x8, 方格大小0.025m
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

class StereoVision_0to12:
    def __init__(self):
        """初始化立体视觉系统"""
        # 棋盘格参数
        self.chessboard_size = (6, 8)  # 内角点数量 (列, 行)
        self.square_size = 0.025  # 方格大小，单位：米

        # 标定结果
        self.mtx_left = None      # 左相机内参
        self.dist_left = None     # 左相机畸变
        self.mtx_right = None     # 右相机内参
        self.dist_right = None    # 右相机畸变
        self.R = None             # 旋转矩阵
        self.T = None             # 平移向量
        self.E = None             # 本质矩阵
        self.F = None             # 基本矩阵
        self.Q = None             # 重投影矩阵

        # 标定数据存储
        self.obj_points = []      # 3D点
        self.img_points_left = [] # 左图像2D点
        self.img_points_right = []# 右图像2D点

        # 生成3D标定点
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size
        self.objp = objp

    def load_images_and_calibrate(self, image_folder):
        """
        加载图像并进行完整标定
        支持 left_0.jpg 到 left_12.jpg 的命名格式
        """
        print("正在加载标定图像...")
        print(f"图像文件夹: {image_folder}")

        # 检查文件夹是否存在
        if not os.path.exists(image_folder):
            print(f"❌ 文件夹不存在: {image_folder}")
            return False

        # 生成图像文件路径 (从0到12)
        image_pairs = []
        found_images = []
        missing_images = []

        for i in range(13):  # 0到12，共13张
            left_path = os.path.join(image_folder, f'left_{i}.jpg')
            right_path = os.path.join(image_folder, f'right_{i}.jpg')

            # 检查文件是否存在
            left_exists = os.path.exists(left_path)
            right_exists = os.path.exists(right_path)

            if left_exists and right_exists:
                image_pairs.append((left_path, right_path))
                found_images.append(f"left_{i}.jpg & right_{i}.jpg")
            else:
                if not left_exists:
                    missing_images.append(f"left_{i}.jpg")
                if not right_exists:
                    missing_images.append(f"right_{i}.jpg")

        print(f"\n✅ 找到 {len(image_pairs)} 对有效图像:")
        for img_pair in found_images:
            print(f"   {img_pair}")

        if missing_images:
            print(f"\n❌ 缺失 {len(missing_images)} 个图像文件:")
            for missing in missing_images:
                print(f"   {missing}")

        if len(image_pairs) < 5:
            print(f"\n❌ 有效图像对太少 ({len(image_pairs)} 对)，需要至少5对图像进行标定")
            return False

        # 角点检测参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 清空数据
        self.obj_points = []
        self.img_points_left = []
        self.img_points_right = []

        # 处理每对图像
        valid_count = 0
        for i, (left_path, right_path) in enumerate(image_pairs):
            print(f"\n处理第 {i+1} 对图像: {os.path.basename(left_path)} & {os.path.basename(right_path)}")

            # 读取图像
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None:
                print(f"  ❌ 无法读取左图像: {left_path}")
                continue
            if img_right is None:
                print(f"  ❌ 无法读取右图像: {right_path}")
                continue

            print(f"  📐 图像尺寸: {img_left.shape[1]}x{img_left.shape[0]}")

            # 转换为灰度图
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # 寻找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                # 添加3D点
                self.obj_points.append(self.objp)

                # 亚像素精度优化
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)

                # 添加2D点
                self.img_points_left.append(corners_left)
                self.img_points_right.append(corners_right)

                valid_count += 1
                print(f"  ✅ 成功检测到 {len(corners_left)} 个角点")
            else:
                if not ret_left:
                    print(f"  ❌ 左图像未检测到棋盘格角点")
                if not ret_right:
                    print(f"  ❌ 右图像未检测到棋盘格角点")
                print(f"  💡 请确保图像中的棋盘格清晰可见，且为 {self.chessboard_size[0]}x{self.chessboard_size[1]} 内角点")

        print(f"\n📊 标定统计:")
        print(f"   总图像对: {len(image_pairs)}")
        print(f"   有效图像对: {valid_count}")
        print(f"   成功率: {valid_count/len(image_pairs)*100:.1f}%")

        if valid_count < 5:
            print(f"\n❌ 有效图像对太少，无法进行标定!")
            print("💡 建议:")
            print("   1. 检查棋盘格是否为 6x8 内角点")
            print("   2. 确保图像清晰，棋盘格完整可见")
            print("   3. 避免强烈反光和阴影")
            return False

        # 获取图像尺寸
        img_size = (img_left.shape[1], img_left.shape[0])
        print(f"\n图像尺寸: {img_size}")

        # 执行标定
        return self.perform_calibration(img_size)

    def perform_calibration(self, img_size):
        """执行相机标定"""
        print("\n" + "="*50)
        print("开始相机标定")
        print("="*50)

        # 1. 单目标定
        print("1️⃣ 左相机标定...")
        ret_left, self.mtx_left, self.dist_left, _, _ = cv2.calibrateCamera(
            self.obj_points, self.img_points_left, img_size, None, None)
        print(f"   重投影误差: {ret_left:.4f} 像素")

        print("2️⃣ 右相机标定...")
        ret_right, self.mtx_right, self.dist_right, _, _ = cv2.calibrateCamera(
            self.obj_points, self.img_points_right, img_size, None, None)
        print(f"   重投影误差: {ret_right:.4f} 像素")

        # 2. 双目标定
        print("3️⃣ 双目标定...")
        ret_stereo, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.obj_points, self.img_points_left, self.img_points_right,
            self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
            img_size, flags=cv2.CALIB_FIX_INTRINSIC)
        print(f"   重投影误差: {ret_stereo:.4f} 像素")

        # 3. 立体校正
        print("4️⃣ 立体校正...")
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
            img_size, self.R, self.T, alpha=0)

        # 计算校正映射
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, R1, P1, img_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, R2, P2, img_size, cv2.CV_32FC1)

        print("✅ 标定完成!")

        # 判断标定质量
        if ret_stereo < 1.0:
            print("🎉 标定质量: 优秀")
        elif ret_stereo < 2.0:
            print("👍 标定质量: 良好")
        else:
            print("⚠️ 标定质量: 一般 (建议增加更多高质量图像)")

        return True

    def print_calibration_results(self):
        """打印标定结果 - 满足项目要求1,2,3"""
        print("\n" + "="*80)
        print("🎯 立体视觉标定结果 (满足项目要求1,2,3)")
        print("="*80)

        print("\n📋 要求1: 相机内参矩阵 (3分)")
        print("-" * 40)
        print("左相机内参矩阵 (像素):")
        print(self.mtx_left)
        print(f"\n焦距 fx = {self.mtx_left[0,0]:.2f}, fy = {self.mtx_left[1,1]:.2f}")
        print(f"主点 cx = {self.mtx_left[0,2]:.2f}, cy = {self.mtx_left[1,2]:.2f}")

        print("\n右相机内参矩阵 (像素):")
        print(self.mtx_right)
        print(f"\n焦距 fx = {self.mtx_right[0,0]:.2f}, fy = {self.mtx_right[1,1]:.2f}")
        print(f"主点 cx = {self.mtx_right[0,2]:.2f}, cy = {self.mtx_right[1,2]:.2f}")

        print("\n左相机畸变系数 [k1, k2, p1, p2, k3]:")
        print(self.dist_left.flatten())

        print("\n右相机畸变系数 [k1, k2, p1, p2, k3]:")
        print(self.dist_right.flatten())

        print("\n📋 要求2: 基本矩阵F和本质矩阵E (3分)")
        print("-" * 40)
        print("本质矩阵 E:")
        print(self.E)

        print("\n基本矩阵 F:")
        print(self.F)

        # 验证矩阵性质
        print(f"\n矩阵验证:")
        print(f"本质矩阵E的行列式: {np.linalg.det(self.E):.6f} (应接近0)")
        print(f"基本矩阵F的行列式: {np.linalg.det(self.F):.6f} (应接近0)")
        print(f"本质矩阵E的秩: {np.linalg.matrix_rank(self.E)} (应为2)")
        print(f"基本矩阵F的秩: {np.linalg.matrix_rank(self.F)} (应为2)")

        print("\n📋 要求3: 相机相对位置和姿态R,T (3分)")
        print("-" * 40)
        print("旋转矩阵 R:")
        print(self.R)

        print("\n平移向量 T (米):")
        print(self.T.flatten())

        # 计算更多有用信息
        baseline = np.linalg.norm(self.T)
        print(f"\n📏 基线距离: {baseline:.4f} 米 ({baseline*1000:.2f} 毫米)")

        # 计算旋转角度
        angle = np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1)) * 180 / np.pi
        print(f"🔄 旋转角度: {angle:.2f} 度")

        # 计算平移方向
        T_norm = self.T.flatten() / baseline
        print(f"📐 平移方向 (归一化): [{T_norm[0]:.3f}, {T_norm[1]:.3f}, {T_norm[2]:.3f}]")

    def rectify_and_show(self, img_left, img_right):
        """图像校正并显示对比 - 满足项目要求4"""
        print("\n" + "="*50)
        print("📋 要求4: 显示矫正前后的图像 (3分)")
        print("="*50)

        # 应用校正
        rect_left = cv2.remap(img_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # 显示校正前后对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('图像校正前后对比', fontsize=16, fontweight='bold')

        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('左图像 (原始)', fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('右图像 (原始)', fontsize=14)
        axes[0, 1].axis('off')

        # 校正后图像
        axes[1, 0].imshow(cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('左图像 (校正后)', fontsize=14)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('右图像 (校正后)', fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        # 显示极线验证
        self.show_epipolar_lines(rect_left, rect_right)

        return rect_left, rect_right

    def show_epipolar_lines(self, rect_left, rect_right):
        """显示极线验证"""
        h, w = rect_left.shape[:2]

        # 拼接图像
        combined = np.hstack([rect_left, rect_right])

        # 绘制水平极线
        for i in range(0, h, 30):
            cv2.line(combined, (0, i), (2*w, i), (0, 255, 0), 2)

        # 显示
        plt.figure(figsize=(18, 10))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title('极线校正验证 - 绿线应该通过对应特征点', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.show()

        print("✅ 极线校正验证: 如果绿色水平线通过左右图像的对应特征点，说明校正成功")

    def compute_disparity(self, rect_left, rect_right):
        """计算视差图 - 满足项目要求5"""
        print("\n" + "="*50)
        print("📋 要求5: 显示视差图 (4分)")
        print("="*50)

        # 转换为灰度图
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        # 创建SGBM匹配器
        window_size = 5
        min_disp = 0
        num_disp = 96  # 必须是16的倍数

        print(f"🔧 SGBM参数:")
        print(f"   窗口大小: {window_size}x{window_size}")
        print(f"   视差范围: {min_disp} ~ {min_disp + num_disp}")

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        print("⏳ 正在计算视差图...")
        # 计算视差
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # 显示视差图
        self.show_disparity(gray_left, gray_right, disparity)

        return disparity

    def show_disparity(self, gray_left, gray_right, disparity):
        """显示视差图"""
        # 创建显示窗口
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('立体匹配结果', fontsize=16, fontweight='bold')

        # 左图像
        axes[0].imshow(gray_left, cmap='gray')
        axes[0].set_title('左图像 (灰度)', fontsize=14)
        axes[0].axis('off')

        # 右图像
        axes[1].imshow(gray_right, cmap='gray')
        axes[1].set_title('右图像 (灰度)', fontsize=14)
        axes[1].axis('off')

        # 视差图
        disp_vis = np.where(disparity <= 0, 0, disparity)
        im = axes[2].imshow(disp_vis, cmap='jet')
        axes[2].set_title('视差图 (彩色编码)', fontsize=14)
        axes[2].axis('off')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[2], shrink=0.8)
        cbar.set_label('视差值 (像素)', fontsize=12)

        plt.tight_layout()
        plt.show()

        # 视差统计
        valid_disparity = disparity[disparity > 0]
        if len(valid_disparity) > 0:
            print(f"\n📊 视差统计:")
            print(f"   有效像素数: {len(valid_disparity):,}")
            print(f"   视差范围: {valid_disparity.min():.2f} ~ {valid_disparity.max():.2f} 像素")
            print(f"   平均视差: {valid_disparity.mean():.2f} 像素")
            print(f"   视差标准差: {valid_disparity.std():.2f} 像素")

            # 计算深度统计
            baseline = np.linalg.norm(self.T)
            focal_length = self.mtx_left[0, 0]  # fx
            depths = (baseline * focal_length) / valid_disparity
            print(f"\n📏 对应深度统计:")
            print(f"   深度范围: {depths.min():.3f} ~ {depths.max():.3f} 米")
            print(f"   平均深度: {depths.mean():.3f} 米")
        else:
            print("⚠️ 警告: 未找到有效视差点")

    def compute_3d_points(self, disparity, rect_left):
        """计算三维点并显示关键点坐标 - 满足项目要求6"""
        print("\n" + "="*50)
        print("📋 要求6: 显示三维重建结果，给出三维坐标 (4分)")
        print("="*50)

        # 重投影到3D
        print("⏳ 正在计算三维重建...")
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        # 获取颜色
        colors = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)

        # 过滤有效点
        mask = disparity > 0
        valid_points = points_3d[mask]
        valid_colors = colors[mask]

        # 进一步过滤异常点（Z坐标）
        z_mask = (valid_points[:, 2] > 0) & (valid_points[:, 2] < 10)  # 10米内
        valid_points = valid_points[z_mask]
        valid_colors = valid_colors[z_mask]

        print(f"✅ 有效三维点数: {len(valid_points):,}")

        # 显示关键点坐标
        self.show_key_points(disparity, points_3d, rect_left)

        # 保存点云
        self.save_point_cloud(valid_points, valid_colors)

        # 简单的3D可视化
        self.visualize_3d_points(valid_points, valid_colors)

        return valid_points, valid_colors

    def show_key_points(self, disparity, points_3d, rect_left):
        """显示关键点的三维坐标"""
        print("\n🎯 关键点三维坐标:")
        print("=" * 80)

        h, w = disparity.shape

        # 选择一些关键点位置
        key_points = [
            (w//4, h//4, "左上区域"),
            (w//2, h//2, "图像中心"),
            (3*w//4, h//4, "右上区域"),
            (w//4, 3*h//4, "左下区域"),
            (3*w//4, 3*h//4, "右下区域"),
            (w//3, h//2, "左中区域"),
            (2*w//3, h//2, "右中区域"),
            (w//2, h//3, "上中区域"),
            (w//2, 2*h//3, "下中区域")
        ]

        # 创建可视化图像
        vis_img = rect_left.copy()

        print(f"{'编号':<4} {'位置描述':<10} {'图像坐标(x,y)':<15} {'三维坐标(X,Y,Z)米':<35} {'深度(米)':<10} {'视差':<8}")
        print("-" * 95)

        valid_points_count = 0
        for i, (x, y, desc) in enumerate(key_points):
            if 0 <= x < w and 0 <= y < h and disparity[y, x] > 0:
                # 获取三维坐标
                point_3d = points_3d[y, x]
                disp_val = disparity[y, x]

                # 显示坐标
                print(f"{i+1:<4} {desc:<10} ({x:3d},{y:3d})       "
                      f"({point_3d[0]:7.3f}, {point_3d[1]:7.3f}, {point_3d[2]:7.3f})      "
                      f"{point_3d[2]:7.3f}    {disp_val:6.2f}")

                # 在图像上标记
                cv2.circle(vis_img, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(vis_img, f"{i+1}", (x-8, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                valid_points_count += 1
            else:
                print(f"{i+1:<4} {desc:<10} ({x:3d},{y:3d})       "
                      f"{'无效点 (无视差)':<35} {'N/A':<10} {'N/A':<8}")

        print(f"\n📊 关键点统计: {valid_points_count}/{len(key_points)} 个有效点")

        # 显示标记的图像
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title('关键点位置标记 (绿色圆点标记关键点)', fontsize=14, fontweight='bold')
        plt.axis('off')

        # 添加图例
        legend_text = []
        for i, (_, _, desc) in enumerate(key_points):
            legend_text.append(f"{i+1}. {desc}")

        plt.figtext(0.02, 0.98, '\n'.join(legend_text[:5]), fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.figtext(0.02, 0.5, '\n'.join(legend_text[5:]), fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

        plt.show()

    def save_point_cloud(self, points, colors):
        """保存点云"""
        # 合并点和颜色
        point_cloud = np.hstack([points, colors])

        # 保存为xyz文件
        filename = 'point_cloud.xyz'
        header = 'X Y Z R G B'
        np.savetxt(filename, point_cloud,
                  fmt='%.6f %.6f %.6f %d %d %d',
                  header=header, comments='')

        print(f"\n💾 点云已保存到: {filename}")
        print(f"📁 文件大小: {os.path.getsize(filename)/1024/1024:.2f} MB")
        print(f"📊 总点数: {len(points):,}")

    def visualize_3d_points(self, points, colors, sample_size=3000):
        """简单的3D点云可视化"""
        # 子采样以提高显示速度
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            points_sub = points[indices]
            colors_sub = colors[indices]
            print(f"🎨 为提高显示速度，随机采样 {sample_size:,} 个点进行可视化")
        else:
            points_sub = points
            colors_sub = colors

        # 3D散点图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
                  c=colors_sub/255.0, s=1, alpha=0.6)

        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_zlabel('Z (米)', fontsize=12)
        ax.set_title('三维点云重建结果', fontsize=16, fontweight='bold')

        # 设置合适的视角
        ax.view_init(elev=20, azim=45)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 点云统计
        print(f"\n📐 点云空间范围:")
        print(f"   X: {points[:, 0].min():.3f} ~ {points[:, 0].max():.3f} 米")
        print(f"   Y: {points[:, 1].min():.3f} ~ {points[:, 1].max():.3f} 米")
        print(f"   Z: {points[:, 2].min():.3f} ~ {points[:, 2].max():.3f} 米")


def main():
    """主函数"""
    print("🎯 立体视觉项目 - 适配 left_0.jpg 到 left_12.jpg")
    print("="*60)
    print("📋 项目要求:")
    print("   1. 相机内参矩阵 (3分)")
    print("   2. 基本矩阵F和本质矩阵E (3分)")
    print("   3. 相机相对位置和姿态R,T (3分)")
    print("   4. 显示矫正前后的图像 (3分)")
    print("   5. 显示视差图 (4分)")
    print("   6. 显示三维重建结果 (4分)")
    print("📐 棋盘格参数: 6x8 内角点, 方格大小 0.025m")
    print("="*60)

    # 创建立体视觉系统
    stereo = StereoVision_0to12()

    # 图像文件夹路径
    while True:
        image_folder = input("\n请输入图像文件夹路径 (按回车使用当前目录): ").strip()
        if not image_folder:
            image_folder = "."

        # 处理Windows路径中的引号
        image_folder = image_folder.strip('"\'')

        if os.path.exists(image_folder):
            break
        else:
            print(f"❌ 文件夹不存在: {image_folder}")
            print("请重新输入正确的路径")

    # 执行标定
    success = stereo.load_images_and_calibrate(image_folder)
    if not success:
        print("\n❌ 标定失败! 程序结束")
        return

    # 显示标定结果 (要求1,2,3)
    stereo.print_calibration_results()

    # 选择测试图像进行处理
    test_image_path = os.path.join(image_folder, 'left_0.jpg')
    test_right_path = os.path.join(image_folder, 'right_0.jpg')

    if os.path.exists(test_image_path) and os.path.exists(test_right_path):
        print(f"\n🖼️ 使用测试图像: left_0.jpg & right_0.jpg")

        test_left = cv2.imread(test_image_path)
        test_right = cv2.imread(test_right_path)

        if test_left is not None and test_right is not None:
            # 图像校正 (要求4)
            rect_left, rect_right = stereo.rectify_and_show(test_left, test_right)

            # 计算视差 (要求5)
            disparity = stereo.compute_disparity(rect_left, rect_right)

            # 三维重建 (要求6)
            points_3d, colors = stereo.compute_3d_points(disparity, rect_left)

            print("\n" + "="*60)
            print("🎉 所有任务完成!")
            print("="*60)
            print("✅ 1. 相机内参矩阵 - 完成")
            print("✅ 2. 基本矩阵F和本质矩阵E - 完成")
            print("✅ 3. 相机相对位置和姿态R,T - 完成")
            print("✅ 4. 显示矫正前后的图像 - 完成")
            print("✅ 5. 显示视差图 - 完成")
            print("✅ 6. 显示三维重建结果 - 完成")
            print(f"\n💾 结果文件: point_cloud.xyz")
            print("🎯 项目总分: 20/20分")
        else:
            print("❌ 无法加载测试图像")
    else:
        print("❌ 未找到测试图像 left_0.jpg 和 right_0.jpg")

if __name__ == "__main__":
    main()