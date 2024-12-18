

using System.Numerics;
using static StarCube.RaMoS.电磁波;

namespace StarCube.RaMoS;//（Radar Modeling System）		


// 添加Debug辅助类
public static class Debug {
	public static void LogVector(string prefix, Vector3D vec) {
		Console.WriteLine($"DEBUG - {prefix}: ({vec.X:F3}, {vec.Y:F3}, {vec.Z:F3})");
	}

	public static void LogValue(string name, double value) {
		Console.WriteLine($"DEBUG - {name}: {value:E3}");
	}
}
// 添加详细的Debug辅助类
public static class DetailedDebug {
	public static void LogAntennaGainCalculation(电磁波 wave, Vector3D targetDirection) {
		Console.WriteLine("\n=== Antenna Gain Calculation Debug ===");
		// 检查波束参数
		Console.WriteLine($"Beamwidth Az: {wave.BeamwidthAzimuth * 180 / Math.PI:F2}°");
		Console.WriteLine($"Beamwidth El: {wave.BeamwidthElevation * 180 / Math.PI:F2}°");
		Console.WriteLine($"Max Gain: {wave.MaxGain:F2} dB");

		// 计算与主瓣方向的夹角
		double angleWithMainLobe = Math.Acos(wave.MainLobeDirection.DotProduct(targetDirection)) * 180 / Math.PI;
		Console.WriteLine($"Angle with main lobe: {angleWithMainLobe:F2}°");

		// 计算方位角和俯仰角
		double azimuth = Math.Atan2(targetDirection.Y, targetDirection.X);
		double elevation = Math.Asin(targetDirection.Z);
		Console.WriteLine($"Target azimuth: {azimuth * 180 / Math.PI:F2}°");
		Console.WriteLine($"Target elevation: {elevation * 180 / Math.PI:F2}°");
	}

	public static void LogTargetUpdate(MovingTarget target, double deltaTime) {
		Console.WriteLine("\n=== Target Update Debug ===");
		Vector3D expectedPosition = new Vector3D(
			target.Position.X + target.Velocity.X * deltaTime,
			target.Position.Y + target.Velocity.Y * deltaTime,
			target.Position.Z + target.Velocity.Z * deltaTime
		);
		Console.WriteLine($"Expected position after update: ({expectedPosition.X:F3}, {expectedPosition.Y:F3}, {expectedPosition.Z:F3})");
		Console.WriteLine($"Delta position expected: ({target.Velocity.X * deltaTime:E3}, {target.Velocity.Y * deltaTime:E3}, {target.Velocity.Z * deltaTime:E3})");
	}

	public static void LogBeamCheck(电磁波 wave, Vector3D targetDirection, double gain) {
		Console.WriteLine("\n=== Beam Coverage Debug ===");
		// 检查-3dB边界
		double minGainThreshold = wave.MaxGain - 3;
		Console.WriteLine($"Gain threshold (-3dB point): {minGainThreshold:F2} dB");
		Console.WriteLine($"Actual gain: {gain:F2} dB");
		Console.WriteLine($"Is target in beam: {gain > minGainThreshold}");
	}
}

public class 电磁波 {
	public double Frequency { get; set; }     // 频率 Hz
	public double Wavelength { get; set; }    // 波长 m
	public double Amplitude { get; set; }     // 振幅
	public double Phase { get; set; }         // 相位 rad
	public Vector3D Direction { get; set; }   // 传播方向
	public Vector3D Position { get; set; }    // 当前位置
	public double Velocity { get; private set; }    // 传播速度 m/s
	public double InitialPower { get; set; }  // 初始功率 W

	// 天线波束相关参数
	public double BeamwidthAzimuth { get; set; }      // 方位角波束宽度(弧度)
	public double BeamwidthElevation { get; set; }    // 俯仰角波束宽度(弧度)
	public double MaxGain { get; set; }               // 最大增益
	public Vector3D MainLobeDirection { get; set; }   // 主瓣方向

	// 极化相关参数
	public Vector3D PolarizationDirection { get; set; }  // 极化方向向量
	public double PolarizationAngle { get; set; }       // 极化角度
	public enum PolarizationType                        // 极化类型
	{
		Linear,         // 线性极化
		Circular,       // 圆极化
		Elliptical,     // 椭圆极化
		Horizontal,  // 添加水平极化
		Vertical     // 可能也需要垂直极化
	}
	public PolarizationType Polarization { get; set; }
	public bool IsRightHanded { get; set; }            // 对于圆/椭圆极化，true表示右旋，false表示左旋

	// 多普勒效应相关属性
	public Vector3D SourceVelocity { get; set; }    // 发射源速度
	public Vector3D ReceiverVelocity { get; set; }  // 接收器速度

	private const double C = 299792458; // 光速 m/s

	// 修改构造函数，添加极化参数
	public 电磁波(double frequency, double amplitude, Vector3D position, Vector3D direction,
					double beamwidthAz, double beamwidthEl, double maxGain,
					PolarizationType polarization = PolarizationType.Linear,
					double polarizationAngle = 0,
					bool isRightHanded = true,
					Vector3D? sourceVelocity = null,    // 使用可空类型
					Vector3D? receiverVelocity = null) {  // 使用可空类型
		Frequency = frequency;
		Amplitude = amplitude;
		Position = position;
		Direction = direction.Normalize();

		// 计算波长
		Wavelength = C / frequency;
		// 初始相位设为0
		Phase = 0;
		// 在自由空间中，传播速度等于光速
		Velocity = C;

		BeamwidthAzimuth = beamwidthAz;
		BeamwidthElevation = beamwidthEl;
		MaxGain = maxGain;
		MainLobeDirection = direction.Normalize();

		Polarization = polarization;
		PolarizationAngle = polarizationAngle;
		IsRightHanded = isRightHanded;

		// 计算极化方向向量（对于线性极化）
		if (polarization == PolarizationType.Linear) {
			// 创建一个垂直于传播方向的极化向量
			Vector3D up = new Vector3D(0, 0, 1);
			PolarizationDirection = Direction.CrossProduct(up).Normalize();
			// 根据极化角度旋转极化方向
			PolarizationDirection = RotateAroundAxis(PolarizationDirection, Direction, polarizationAngle);
		}

		SourceVelocity = sourceVelocity ?? new Vector3D(0, 0, 0);
		ReceiverVelocity = receiverVelocity ?? new Vector3D(0, 0, 0);
	}

	// 更新电磁波的位置和相位
	public void Update(double deltaTime) {
		// 更新位置
		Position += Direction * (Velocity * deltaTime);

		// 更新相位
		double deltaPhase = 2 * Math.PI * Frequency * deltaTime;
		Phase = (Phase + deltaPhase) % (2 * Math.PI);

		// 计算自由空间损耗
		UpdateAmplitude(deltaTime);
	}

	// 计算自由空间损耗
	private void UpdateAmplitude(double deltaTime) {
		// 使用自由空间传播损耗公式
		// Power ~ 1/r^2，其中r是传播距离
		double distance = Velocity * deltaTime;
		if (distance > 0) {
			double r = Position.Magnitude();
			if (r > 0) {
				// 自由空间损耗
				Amplitude *= 1.0 / (4 * Math.PI * r * r);
			}
		}
	}

	// 获取当前点的场强
	// 第一次重写GetFieldStrength方法，考虑方向性增益
	// 第二次修改GetFieldStrength方法，考虑极化损耗
	// 第三次修改GetFieldStrength方法，考虑大气衰减
	// 第四次修改GetFieldStrength方法，考虑多普勒效应
	public double GetFieldStrength(Vector3D observationPoint, Vector3D targetVelocity,
								 AtmosphericCondition atmosphere, double time) {
		Vector3D directionToPoint = (observationPoint - Position).Normalize();
		double gain = CalculateAntennaGain(directionToPoint);

		// 计算距离
		double r = (observationPoint - Position).Magnitude();
		if (r == 0) return 0;

		// 计算基础场强，加入多普勒相位
		double k = 2 * Math.PI / Wavelength;
		double dopplerPhase = CalculateDopplerPhase(observationPoint, targetVelocity, time);
		double baseField = Amplitude * Math.Cos(
			2 * Math.PI * Frequency * (r / Velocity) - k * r + Phase + dopplerPhase);

		// 计算极化损耗
		Vector3D surfaceNormal = new Vector3D(0, 0, 1);
		double polarizationLoss = CalculatePolarizationLoss(directionToPoint, surfaceNormal);

		// 计算大气衰减
		double atmosphericLoss = CalculateAtmosphericAttenuation(atmosphere, r);

		// 应用所有损耗
		return baseField * Math.Sqrt(gain) * polarizationLoss * atmosphericLoss;
	}
	// 第一次修改天线增益计算方法
	// 第二次修改天线增益计算方法
	public double CalculateAntennaGain(Vector3D targetDirection) {
		// 归一化方向向量
		Vector3D normalizedDirection = targetDirection.Normalize();
		Vector3D normalizedMainLobe = MainLobeDirection.Normalize();

		// 计算与主瓣方向的夹角（弧度）
		double angle = Math.Acos(normalizedMainLobe.DotProduct(normalizedDirection));

		// Debug夹角信息
		Debug.LogValue("Angle with main lobe (degrees)", angle * 180 / Math.PI);

		// 如果夹角大于波束宽度的1.5倍，返回最小增益
		if (angle > BeamwidthAzimuth * 1.5) {
			return -30; // 设置一个最小增益值
		}

		// 使用高斯函数计算增益，但调整衰减系数
		double relativeAngle = angle / BeamwidthAzimuth;
		double gainFactor = Math.Exp(-1.386 * Math.Pow(relativeAngle, 2)); // 减小衰减系数

		// 将相对增益转换为dB
		double gainDB = MaxGain * gainFactor;

		// Debug增益计算
		Debug.LogValue("Relative angle", relativeAngle);
		Debug.LogValue("Gain factor", gainFactor);
		Debug.LogValue("Final gain (dB)", gainDB);

		return gainDB;
	}
	// 获取-3dB波束范围的边界点（用于可视化）
	public List<Vector3D> GetBeamBoundaryPoints(int numberOfPoints = 36) {
		List<Vector3D> boundaryPoints = new List<Vector3D>();

		for (int i = 0; i < numberOfPoints; i++) {
			double angle = 2 * Math.PI * i / numberOfPoints;

			// 计算椭圆边界点
			double x = BeamwidthAzimuth * Math.Cos(angle);
			double y = BeamwidthElevation * Math.Sin(angle);

			// 转换到主瓣方向的坐标系
			Vector3D point = TransformToMainLobeDirection(new Vector3D(x, y, 1));
			boundaryPoints.Add(Position + point);
		}

		return boundaryPoints;
	}
	// 将点从波束坐标系转换到全局坐标系
	private Vector3D TransformToMainLobeDirection(Vector3D point) {
		// 计算从z轴到主瓣方向的旋转矩阵
		Vector3D zAxis = new Vector3D(0, 0, 1);
		Vector3D rotationAxis = zAxis.CrossProduct(MainLobeDirection);
		double rotationAngle = Math.Acos(zAxis.DotProduct(MainLobeDirection));

		// 实现旋转矩阵变换
		// 这里使用Rodrigues旋转公式
		if (rotationAxis.Magnitude() < 1e-6) {
			return point;
		}

		rotationAxis = rotationAxis.Normalize();

		// Rodrigues旋转公式
		return point * Math.Cos(rotationAngle) +
			   rotationAxis.CrossProduct(point) * Math.Sin(rotationAngle) +
			   rotationAxis * rotationAxis.DotProduct(point) * (1 - Math.Cos(rotationAngle));
	}
	// 计算极化损耗
	public double CalculatePolarizationLoss(Vector3D incidentDirection, Vector3D surfaceNormal) {
		switch (Polarization) {
			case PolarizationType.Linear:
				return CalculateLinearPolarizationLoss(incidentDirection, surfaceNormal);
			case PolarizationType.Circular:
				return CalculateCircularPolarizationLoss(incidentDirection, surfaceNormal);
			case PolarizationType.Elliptical:
				return CalculateEllipticalPolarizationLoss(incidentDirection, surfaceNormal);
			default:
				return 1.0;
		}
	}

	private double CalculateLinearPolarizationLoss(Vector3D incidentDirection, Vector3D surfaceNormal) {
		// 计算入射角
		double incidenceAngle = Math.Acos(incidentDirection.DotProduct(surfaceNormal));

		// 计算极化方向在表面的投影
		Vector3D projectedPolarization = PolarizationDirection -
			surfaceNormal * PolarizationDirection.DotProduct(surfaceNormal);

		if (projectedPolarization.Magnitude() < 1e-6)
			return 0;

		projectedPolarization = projectedPolarization.Normalize();

		// 计算极化损耗系数 (使用Fresnel方程)
		double n1 = 1.0; // 空气的折射率
		double n2 = 1.5; // 介质的折射率（这个值应该从材料属性中获取）

		// 计算透射角 (使用斯涅尔定律)
		double transmissionAngle = Math.Asin(n1 * Math.Sin(incidenceAngle) / n2);

		// 计算平行和垂直分量的反射系数
		double rs = Math.Pow(
			(n1 * Math.Cos(incidenceAngle) - n2 * Math.Cos(transmissionAngle)) /
			(n1 * Math.Cos(incidenceAngle) + n2 * Math.Cos(transmissionAngle)), 2);

		double rp = Math.Pow(
			(n2 * Math.Cos(incidenceAngle) - n1 * Math.Cos(transmissionAngle)) /
			(n2 * Math.Cos(incidenceAngle) + n1 * Math.Cos(transmissionAngle)), 2);

		// 计算总的极化损耗
		return (rs + rp) / 2;
	}

	private double CalculateCircularPolarizationLoss(Vector3D incidentDirection, Vector3D surfaceNormal) {
		// 圆极化的损耗计算
		// 对于理想的圆极化，损耗相对较小且较为均匀
		double incidenceAngle = Math.Acos(incidentDirection.DotProduct(surfaceNormal));

		// 圆极化的轴向比(Axial Ratio)为1
		double axialRatio = 1.0;

		// 考虑旋向
		double handednessLoss = IsRightHanded ? 1.0 : 0.5; // 假设接收端为右旋

		return Math.Cos(incidenceAngle) * handednessLoss;
	}

	private double CalculateEllipticalPolarizationLoss(Vector3D incidentDirection, Vector3D surfaceNormal) {
		// 椭圆极化可以看作是线性极化和圆极化的组合
		double linearLoss = CalculateLinearPolarizationLoss(incidentDirection, surfaceNormal);
		double circularLoss = CalculateCircularPolarizationLoss(incidentDirection, surfaceNormal);

		// 假设椭圆极化度为0.7（可以根据实际情况调整）
		double ellipticalFactor = 0.7;

		return linearLoss * (1 - ellipticalFactor) + circularLoss * ellipticalFactor;
	}

	// 绕轴旋转向量的辅助方法
	private Vector3D RotateAroundAxis(Vector3D vector, Vector3D axis, double angle) {
		// 使用Rodrigues旋转公式
		axis = axis.Normalize();
		return vector * Math.Cos(angle) +
			   axis.CrossProduct(vector) * Math.Sin(angle) +
			   axis * axis.DotProduct(vector) * (1 - Math.Cos(angle));
	}



	// 计算大气衰减
	public double CalculateAtmosphericAttenuation(
		AtmosphericCondition atmosphere,
		double distance) // 距离(m)
	{
		double freqGHz = Frequency / 1e9; // 转换为GHz

		// 计算各种衰减
		double oxygenAttenuation = CalculateOxygenAttenuation(
			freqGHz,
			atmosphere.Temperature,
			atmosphere.Pressure);

		double waterVaporAttenuation = CalculateWaterVaporAttenuation(
			freqGHz,
			atmosphere.Temperature,
			atmosphere.WaterVaporDensity);

		double rainAttenuation = CalculateRainAttenuation(
			freqGHz,
			atmosphere.RainRate);

		// 总衰减(dB) = 比衰减(dB/km) * 距离(km)
		double totalAttenuation =
			(oxygenAttenuation + waterVaporAttenuation + rainAttenuation) *
			(distance / 1000.0);

		// 转换dB为倍数
		return Math.Pow(10, -totalAttenuation / 10.0);
	}

	// 计算氧气吸收(dB/km)
	private double CalculateOxygenAttenuation(
		double freqGHz,
		double temperature,
		double pressure) {
		// 使用ITU-R P.676-12模型简化版
		double theta = 300.0 / temperature;

		// 氧气共振频率
		double[] f0 = { 50.474214, 50.987745, 51.503360, 52.021429, 52.542418, 53.066934,
					   53.595775, 54.130025, 54.671180, 55.221384, 55.783815, 56.264774,
					   56.363399, 56.968211, 57.612486, 58.323877, 58.446588, 59.164204,
					   59.590983, 60.306056, 60.434778, 61.150562, 61.800158, 62.411220,
					   62.486253, 62.997984, 63.568526, 64.127775, 64.678910, 65.224078,
					   65.764779, 66.302096, 66.836834, 67.369601, 67.900868, 68.431006,
					   68.960312, 118.750334 };

		double totalAttenuation = 0;

		foreach (double f in f0) {
			// 线宽计算
			double width = 1.0 + 1.67e-3 * pressure * Math.Pow(theta, 0.85);

			// 计算吸收系数
			double absorption = width * pressure * Math.Pow(theta, 3) *
							  Math.Exp(-1.0 * Math.Abs(freqGHz - f));

			totalAttenuation += absorption;
		}

		return totalAttenuation * 0.182; // 转换为dB/km
	}

	// 计算水汽吸收(dB/km)
	private double CalculateWaterVaporAttenuation(
		double freqGHz,
		double temperature,
		double waterVaporDensity) {
		// 使用ITU-R P.676-12模型简化版
		double theta = 300.0 / temperature;

		// 水汽共振频率
		double[] f0 = { 22.235080, 67.803960, 119.995940, 183.310087, 321.225630,
					   325.152888, 336.227764, 380.197353, 390.134508, 437.346667,
					   439.150812, 443.018295, 448.001075, 470.888947, 474.689127,
					   488.491133, 503.568532, 504.482692, 547.676440, 552.020960,
					   556.935985, 620.700807, 645.766085, 658.145275, 752.033227,
					   841.053973, 859.962313, 899.306675, 902.616173, 906.207325,
					   916.171582, 923.118427, 970.315022, 987.926764 };

		double totalAttenuation = 0;

		foreach (double f in f0) {
			// 线宽计算
			double width = 2.85 * waterVaporDensity * Math.Pow(theta, 0.626);

			// 计算吸收系数
			double absorption = width * waterVaporDensity * Math.Pow(theta, 3.5) *
							  Math.Exp(-1.0 * Math.Abs(freqGHz - f));

			totalAttenuation += absorption;
		}

		return totalAttenuation * 0.182; // 转换为dB/km
	}

	// 计算降雨衰减(dB/km)
	private double CalculateRainAttenuation(double freqGHz, double rainRate) {
		// 使用ITU-R P.838-3模型简化版
		if (rainRate <= 0) return 0;

		// 频率相关系数
		double k = 0.0001 * freqGHz;
		double alpha = 1.0;

		if (freqGHz < 2.9) {
			k = 0.0000387 * freqGHz;
			alpha = 0.912;
		} else if (freqGHz < 54) {
			k = 0.00158 * freqGHz;
			alpha = 1.076;
		} else if (freqGHz < 180) {
			k = 0.00454 * freqGHz;
			alpha = 1.226;
		}

		// 计算降雨衰减
		return k * Math.Pow(rainRate, alpha);
	}
	// 计算多普勒频移
	public double CalculateDopplerShift(Vector3D targetPosition, Vector3D targetVelocity) {
		// 计算方向向量
		Vector3D directionToTarget = (targetPosition - Position).Normalize();

		// 计算径向速度分量
		double sourceRadialVelocity = SourceVelocity.DotProduct(directionToTarget);
		double targetRadialVelocity = targetVelocity.DotProduct(directionToTarget);
		double receiverRadialVelocity = ReceiverVelocity.DotProduct(directionToTarget);

		// 使用相对论多普勒公式
		double beta_s = sourceRadialVelocity / C;
		double beta_t = targetRadialVelocity / C;
		double beta_r = receiverRadialVelocity / C;

		// 计算多普勒频移
		double shiftedFrequency = Frequency *
			Math.Sqrt((1 - beta_s * beta_s) * (1 - beta_r * beta_r)) /
			((1 + beta_r) * (1 + beta_s));

		return shiftedFrequency - Frequency;
	}

	// 计算考虑多普勒效应的相位变化
	private double CalculateDopplerPhase(Vector3D targetPosition, Vector3D targetVelocity, double deltaTime) {
		double dopplerShift = CalculateDopplerShift(targetPosition, targetVelocity);
		return 2 * Math.PI * dopplerShift * deltaTime;
	}
}

public struct Vector3D {
	public double X { get; set; }
	public double Y { get; set; }
	public double Z { get; set; }

	public Vector3D(double x, double y, double z) {
		X = x;
		Y = y;
		Z = z;
	}

	public double Magnitude() {
		return Math.Sqrt(X * X + Y * Y + Z * Z);
	}

	public Vector3D Normalize() {
		double mag = Magnitude();
		if (mag == 0) return new Vector3D(0, 0, 0);
		return new Vector3D(X / mag, Y / mag, Z / mag);
	}

	public double DotProduct(Vector3D other) {
		return X * other.X + Y * other.Y + Z * other.Z;
	}

	public Vector3D CrossProduct(Vector3D other) {
		return new Vector3D(
			Y * other.Z - Z * other.Y,
			Z * other.X - X * other.Z,
			X * other.Y - Y * other.X
		);
	}

	// 添加基本运算符重载
	public static Vector3D operator +(Vector3D a, Vector3D b)
		=> new Vector3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

	public static Vector3D operator -(Vector3D a, Vector3D b)
		=> new Vector3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

	public static Vector3D operator *(Vector3D a, double b)
		=> new Vector3D(a.X * b, a.Y * b, a.Z * b);
}

public abstract class 仿真物体 {

	public Guid Id { get; } = Guid.NewGuid(); // 添加唯一标识符
	public Vector3D Position { get; set; }
	public 材料 Material { get; set; }
	// 包围盒用于碰撞检测
	public BoundingBox Bounds { get; set; }

	public double RCS { get; set; }            // 雷达散射截面积
											   // 记录与其他目标的相互作用
	protected Dictionary<Guid, InteractionInfo> InteractionWithOthers
		= new Dictionary<Guid, InteractionInfo>();

	// 检查是否遮挡其他目标
	protected virtual bool IsBlockingPath(仿真物体 other) {
		// 实现射线追踪以检查遮挡
		return false; // 基础实现
	}

	// 计算遮挡因子
	protected virtual double CalculateBlockingFactor(仿真物体 other) {
		if (!IsBlockingPath(other)) return 1.0;

		// 基于材料属性和几何关系计算遮挡因子
		double distance = (Position - other.Position).Magnitude();
		return Math.Exp(-distance / 1000.0); // 简化模型
	}

	// 修改后的交互计算方法，考虑多目标效应
	public abstract InteractionResult CalculateInteraction(
		电磁波 wave,
		IEnumerable<仿真物体> otherObjects);
	// 计算与其他目标的相互作用
	public virtual void CalculateInteractionWithOther(仿真物体 other) {
		double distance = (Position - other.Position).Magnitude();
		bool isBlocking = IsBlockingPath(other);

		InteractionWithOthers[other.Id] = new InteractionInfo {
			Distance = distance,
			IsBlocking = isBlocking,
			BlockingFactor = CalculateBlockingFactor(other)
		};
	}
}
public class InteractionInfo {
	public double Distance { get; set; }
	public bool IsBlocking { get; set; }
	public double BlockingFactor { get; set; }
}
public struct BoundingBox {
	public Vector3D Min { get; set; }
	public Vector3D Max { get; set; }

	public BoundingBox(Vector3D min, Vector3D max) {
		Min = min;
		Max = max;
	}
}

public class 材料 {
	public double RelativePermittivity { get; set; }   // 相对介电常数
	public double Conductivity { get; set; }           // 电导率
	public double RoughnessFactor { get; set; }        // 表面粗糙度
}

public class 仿真环境 {
	private List<电磁波> activeWaves;
	private List<仿真物体> objects;
	private double timeStep; // 仿真时间步长
	private AtmosphericCondition atmosphere;
	private double simulationTime = 0;  // 仿真时间
	private List<InteractionResult> interactions = new List<InteractionResult>();
	private List<Clutter> clutters = new List<Clutter>();
	private Random random = new Random();
	private const double DEFAULT_MAX_GAIN = -3.0; // 添加默认增益值
	public double TimeStep => timeStep;

	public 仿真环境(double timeStep = 1e-9, AtmosphericCondition atmosphere = null) {
		this.timeStep = timeStep;
		this.atmosphere = atmosphere ?? new AtmosphericCondition();
		activeWaves = new List<电磁波>();
		objects = new List<仿真物体>();
	}

	// 添加这个方法
	public void AddObject(仿真物体 obj) {
		objects.Add(obj);
	}

	// 添加这个方法来移除物体
	public void RemoveObject(仿真物体 obj) {
		objects.Remove(obj);
	}



	public void Update() {
		simulationTime += timeStep;

		// 更新所有运动目标
		foreach (var obj in objects) {
			if (obj is MovingTarget movingTarget) {
				movingTarget.Update(timeStep);
			}
		}

		// 更新所有电磁波
		UpdateWaves();

		// 处理交互
		ProcessInteractions();
	}

	public void AddWave(电磁波 wave) {
		activeWaves.Add(wave);
	}

	public void UpdateWaves() {
		// 更新所有电磁波
		foreach (var wave in activeWaves.ToList()) {
			wave.Update(timeStep);

			// 检查波是否已经传播超出有效范围
			if (IsOutOfBounds(wave.Position)) {
				activeWaves.Remove(wave);
				continue;
			}

			// 获取波束覆盖范围内的目标
			var objectsInBeam = GetObjectsInBeam(wave);

			// 处理每个目标的散射
			foreach (var obj in objectsInBeam) {
				// 计算目标散射
				var result = obj.CalculateInteraction(wave, objectsInBeam);
				if (result != null) {
					interactions.Add(result);
				}
			}
		}

		// 更新所有运动目标
		foreach (var obj in objects) {
			if (obj is MovingTarget movingTarget) {
				movingTarget.Update(timeStep);
			}
		}
	}

	private bool IsOutOfBounds(Vector3D position) {
		// 根据仿真空间的边界进行检查
		// 这里需要根据实际的仿真空间大小来设定边界
		const double BOUNDARY = 1000.0; // 示例边界值
		return Math.Abs(position.X) > BOUNDARY ||
			   Math.Abs(position.Y) > BOUNDARY ||
			   Math.Abs(position.Z) > BOUNDARY;
	}

	// 计算空间中某点的总场强
	// 第一次修改计算总场强的方法
	// 第二次修改计算总场强的方法，考虑多普勒效应
	// 第三次修改场强计算方法，考虑多目标叠加效应
	public double CalculateTotalFieldStrength(Vector3D point) {
		Complex totalField = new Complex(0, 0);

		foreach (var interaction in interactions) {
			// 计算从目标到观测点的传播
			Vector3D targetToPoint = point - interaction.SourceObject.Position;
			double distance = targetToPoint.Magnitude();

			// 计算相位
			double phase = interaction.PhaseShift +
						 2 * Math.PI * distance / activeWaves[0].Wavelength;

			// 计算幅度
			double amplitude = Math.Sqrt(interaction.ReflectedPower);

			// 添加多径效应
			foreach (var multiPath in interaction.MultiPathEffects) {
				phase += multiPath.Value;
			}

			// 转换为复数并叠加
			Complex fieldComponent = Complex.FromPolarCoordinates(
				amplitude,
				phase
			);

			totalField += fieldComponent;
		}

		// 杂波回波
		foreach (var clutter in clutters) {
			foreach (var wave in activeWaves) {
				double clutterReturn = clutter.CalculateClutterReturn(wave, atmosphere);
				double clutterDoppler = clutter.CalculateClutterDoppler(wave);

				// 计算杂波相位
				Vector3D clutterToPoint = point - clutter.Position;
				double distance = clutterToPoint.Magnitude();
				double phase = 2 * Math.PI * distance / wave.Wavelength +
							 2 * Math.PI * clutterDoppler * simulationTime;

				Complex clutterField = Complex.FromPolarCoordinates(
					Math.Sqrt(clutterReturn),
					phase
				);

				totalField += clutterField;
			}
		}

		return totalField.Magnitude;
	}

	// 获取波束覆盖范围内的物体
	// 第一次修改GetObjectsInBeam方法
	public List<仿真物体> GetObjectsInBeam(电磁波 wave) {
		List<仿真物体> objectsInBeam = new List<仿真物体>();

		foreach (var obj in objects) {
			Vector3D directionToObject = (obj.Position - wave.Position).Normalize();
			double gain = wave.CalculateAntennaGain(directionToObject);

			// 如果增益高于-3dB点，认为目标在波束内
			if (gain > wave.MaxGain - 3) {
				objectsInBeam.Add(obj);
			}
		}

		return objectsInBeam;
	}

	// 更新大气条件
	public void UpdateAtmosphere(AtmosphericCondition newAtmosphere) {
		this.atmosphere = newAtmosphere;
	}
	public void ProcessInteractions() {
		interactions.Clear();

		foreach (var wave in activeWaves) {
			// 获取波束覆盖范围内的所有目标
			var objectsInBeam = GetObjectsInBeam(wave);

			// 计算目标间的相互作用
			foreach (var obj1 in objectsInBeam) {
				foreach (var obj2 in objectsInBeam) {
					if (obj1.Id != obj2.Id) {
						obj1.CalculateInteractionWithOther(obj2);
					}
				}
			}

			// 计算每个目标的散射
			foreach (var obj in objectsInBeam) {
				var interaction = obj.CalculateInteraction(wave, objectsInBeam);
				interactions.Add(interaction);
			}
		}
	}
	private Vector3D GetNearestTargetVelocity(Vector3D point) {
		MovingTarget nearestTarget = null;
		double minDistance = double.MaxValue;

		foreach (var obj in objects) {
			if (obj is MovingTarget target) {
				double distance = (target.Position - point).Magnitude();
				if (distance < minDistance) {
					minDistance = distance;
					nearestTarget = target;
				}
			}
		}

		return nearestTarget?.Velocity ?? new Vector3D(0, 0, 0);
	}

	// 分析多目标场景
	public MultiTargetAnalysisResult AnalyzeMultiTargetScene() {
		// 检查是否有交互数据
		if (!interactions.Any()) {
			return new MultiTargetAnalysisResult {
				Interactions = new List<InteractionResult>(),
				TotalScatteredPower = 0,
				MaxDopplerShift = 0,
				InteractionCount = 0
			};
		}

		return new MultiTargetAnalysisResult {
			Interactions = interactions,
			TotalScatteredPower = interactions.Sum(i => i.ReflectedPower),
			MaxDopplerShift = interactions.Max(i => Math.Abs(i.DopplerShift)),
			InteractionCount = interactions.Count
		};
	}

	public void GenerateClutter(
		string clutterType,
		Vector3D center,
		double radius,
		int numberOfPoints) {
		for (int i = 0; i < numberOfPoints; i++) {
			// 在指定区域内随机生成杂波点
			Vector3D position = GenerateRandomPosition(center, radius);

			Clutter clutter;
			switch (clutterType.ToLower()) {
				case "ground":
					clutter = new GroundClutter {
						Position = position,
						TerrainType = "grass",  // 可以根据需要修改
						Roughness = 0.1,
						TerrainSlope = 0
					};
					break;

				case "volume":
					clutter = new VolumeClutter {
						Position = position,
						ClutterType = "rain",
						Density = 0.001,
						ParticleSize = 0.002
					};
					break;

				case "sea":
					clutter = new SeaClutter {
						Position = position,
						SeaState = 3,
						WindSpeed = 5.0,
						WindDirection = 0
					};
					break;

				default:
					throw new ArgumentException("Unknown clutter type");
			}

			clutters.Add(clutter);
		}
	}
	private Vector3D GenerateRandomPosition(Vector3D center, double radius) {
		double r = radius * Math.Sqrt(random.NextDouble());
		double theta = random.NextDouble() * 2 * Math.PI;

		return new Vector3D(
			center.X + r * Math.Cos(theta),
			center.Y + r * Math.Sin(theta),
			center.Z
		);
	}

}
public class MultiTargetAnalysisResult {
	public List<InteractionResult> Interactions { get; set; }
	public double TotalScatteredPower { get; set; }
	public double MaxDopplerShift { get; set; }
	public int InteractionCount { get; set; }
}
public class AtmosphericCondition {
	public double Temperature { get; set; }      // 温度(K)
	public double Pressure { get; set; }         // 气压(hPa)
	public double Humidity { get; set; }         // 相对湿度(%)
	public double RainRate { get; set; }         // 降雨率(mm/h)
	public double WaterVaporDensity { get; set; }// 水汽密度(g/m³)
	public double RelativePermittivity { get; set; } // 添加相对介电常数

	public AtmosphericCondition(
		double temperature = 288.15,  // 15℃
		double pressure = 1013.25,    // 标准大气压
		double humidity = 60,         // 60%相对湿度
		double rainRate = 0,          // 无雨
		double waterVaporDensity = 7.5,// 标准水汽密度
		double relativePermittivity = 1.0) {
		Temperature = temperature;
		Pressure = pressure;
		Humidity = humidity;
		RainRate = rainRate;
		WaterVaporDensity = waterVaporDensity;
		RelativePermittivity = relativePermittivity;
	}
}

public class MovingTarget : 仿真物体 {
	public Vector3D Velocity { get; set; }
	public Vector3D Acceleration { get; set; }
	public double CrossSection { get; set; }  // 目标横截面积

	public MovingTarget(Vector3D position, Vector3D velocity, Vector3D? acceleration = null) {
		Position = position;
		Velocity = velocity;
		Acceleration = acceleration ?? new Vector3D(0, 0, 0);
	}

	public void Update(double deltaTime) {
		// 更新速度
		Velocity += Acceleration * deltaTime;

		// 更新位置
		Position = new Vector3D(
			Position.X + Velocity.X * deltaTime,
			Position.Y + Velocity.Y * deltaTime,
			Position.Z + Velocity.Z * deltaTime
		);
	}

	// 重写交互计算方法
	// 第一次修改交互计算方法
	// 第二次修改MovingTarget类中的计算交互方法
	public override InteractionResult CalculateInteraction(
		电磁波 wave,
		IEnumerable<仿真物体> otherObjects) {

		// 计算到波源的方向
		Vector3D directionFromWave = (Position - wave.Position).Normalize();

		// 计算天线增益
		double antennaGain = wave.CalculateAntennaGain(directionFromWave);
		Debug.LogValue("Antenna Gain for Interaction", antennaGain);

		// 即使增益较小也计算交互，只要不是最小增益
		if (antennaGain > -30) {
			// 计算多普勒频移
			double dopplerShift = wave.CalculateDopplerShift(Position, Velocity);

			// 计算距离
			double distance = (Position - wave.Position).Magnitude();

			// 计算入射功率，考虑空间衰减和天线增益（转换dB为线性值）
			double gainLinear = Math.Pow(10, antennaGain / 10);
			double incidentPower = wave.InitialPower * gainLinear /
								 (4 * Math.PI * distance * distance);

			Debug.LogValue("Incident Power", incidentPower);

			// 计算反射功率，考虑RCS
			double reflectedPower = incidentPower * RCS /
								  (4 * Math.PI * distance * distance);

			Debug.LogValue("Reflected Power", reflectedPower);

			// 计算相位
			double phaseShift = 4 * Math.PI * distance / wave.Wavelength;

			return new InteractionResult {
				DopplerShift = dopplerShift,
				ReflectedPower = reflectedPower,
				PhaseShift = phaseShift,
				SourceObject = this
			};
		}

		return null;
	}

	private double CalculateReflectedPower(电磁波 wave) {
		// 使用雷达方程计算反射功率
		Vector3D directionToTarget = (Position - wave.Position).Normalize();
		double distance = (Position - wave.Position).Magnitude();

		// 计算入射功率，考虑天线增益和空间损耗
		double incidentPower = wave.InitialPower *
			wave.CalculateAntennaGain(directionToTarget) /
			(4 * Math.PI * distance * distance);

		// 计算反射功率，考虑RCS
		return incidentPower * RCS / (4 * Math.PI * distance * distance);
	}

	private double CalculatePhaseShift(电磁波 wave) {
		double distance = (Position - wave.Position).Magnitude();
		return 4 * Math.PI * distance / wave.Wavelength;
	}
}

public class InteractionResult {
	public double DopplerShift { get; set; }
	public double ReflectedPower { get; set; }
	public double PhaseShift { get; set; }
	public 仿真物体 SourceObject { get; set; }
	public Dictionary<Guid, double> MultiPathEffects { get; set; }
		= new Dictionary<Guid, double>();
}

public abstract class Clutter {
	public Vector3D Position { get; set; }
	public double RCS { get; set; }          // 杂波等效散射截面积
	public Vector3D Velocity { get; set; }   // 杂波运动速度
	public double Correlation { get; set; }  // 时间相关性

	// 计算杂波回波
	public abstract double CalculateClutterReturn(电磁波 wave, AtmosphericCondition atmosphere);

	// 计算杂波多普勒频移
	public abstract double CalculateClutterDoppler(电磁波 wave);
}

// 地面杂波
public class GroundClutter : Clutter {
	public double Roughness { get; set; }        // 地表粗糙度
	public double TerrainSlope { get; set; }     // 地形坡度
	public string TerrainType { get; set; }      // 地形类型(如: 草地、沙漠、城市等)

	private readonly Dictionary<string, double> TerrainReflectivity = new Dictionary<string, double>
	{
		{"grass", -15.0},      // dB
        {"desert", -25.0},
		{"urban", -5.0},
		{"forest", -10.0},
		{"snow", -20.0}
	};

	public override double CalculateClutterReturn(电磁波 wave, AtmosphericCondition atmosphere) {
		// 计算入射角
		Vector3D directionToClutter = (Position - wave.Position).Normalize();
		double incidenceAngle = Math.Acos(directionToClutter.DotProduct(new Vector3D(0, 0, 1)));

		// 获取地形反射率
		double reflectivity = TerrainReflectivity.ContainsKey(TerrainType)
			? TerrainReflectivity[TerrainType]
			: -15.0;

		// 考虑地形坡度影响
		double slopeEffect = Math.Cos(TerrainSlope);

		// 考虑粗糙度影响 (使用改进的Ament模型)
		double k = 2 * Math.PI / wave.Wavelength;
		double roughnessEffect = Math.Exp(-8 * Math.Pow(k * Roughness * Math.Sin(incidenceAngle), 2));

		// 计算归一化杂波截面积 (sigma0)
		double sigma0 = Math.Pow(10, reflectivity / 10) * slopeEffect * roughnessEffect;

		return sigma0;
	}

	public override double CalculateClutterDoppler(电磁波 wave) {
		// 地面杂波多普勒主要来自平台运动
		return wave.CalculateDopplerShift(Position, Velocity);
	}
}

// 体杂波
public class VolumeClutter : Clutter {
	public double Density { get; set; }          // 散射体密度
	public double ParticleSize { get; set; }     // 散射体尺寸
	public string ClutterType { get; set; }      // 体杂波类型(如: 雨、雪、云等)

	private readonly Dictionary<string, (double density, double size)> DefaultParameters =
		new Dictionary<string, (double density, double size)>
	{
		{"rain", (0.001, 0.002)},    // 密度(g/m³), 尺寸(m)
        {"snow", (0.1, 0.005)},
		{"cloud", (0.5, 0.0001)},
		{"dust", (0.01, 0.0001)}
	};

	public override double CalculateClutterReturn(电磁波 wave, AtmosphericCondition atmosphere) {
		// 使用Rayleigh散射模型
		double k = 2 * Math.PI / wave.Wavelength;

		// 计算单个散射体的RCS
		double particleRCS = Math.Pow(k, 4) * Math.Pow(ParticleSize, 6) *
			Math.Pow((atmosphere.RelativePermittivity - 1) / (atmosphere.RelativePermittivity + 2), 2);

		// 考虑体积内散射体数量
		double volume = 1.0; // 考虑单位体积
		double numberOfParticles = Density * volume;

		// 总的体杂波RCS
		return particleRCS * numberOfParticles;
	}

	public override double CalculateClutterDoppler(电磁波 wave) {
		// 考虑大气运动和湍流
		Vector3D effectiveVelocity = Velocity + new Vector3D(
			GaussianRandom(0, 1),  // 添加随机湍流
			GaussianRandom(0, 1),
			GaussianRandom(0, 1)
		);

		return wave.CalculateDopplerShift(Position, effectiveVelocity);
	}

	private double GaussianRandom(double mean, double stdDev) {
		Random rand = new Random();
		double u1 = 1.0 - rand.NextDouble();
		double u2 = 1.0 - rand.NextDouble();
		double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
		return mean + stdDev * randStdNormal;
	}
}

// 海面杂波
public class SeaClutter : Clutter {
	public double WaveHeight { get; set; }       // 海浪高度
	public double WindSpeed { get; set; }        // 风速
	public double WindDirection { get; set; }    // 风向
	public int SeaState { get; set; }           // 海况等级(0-9)

	private readonly Dictionary<int, (double height, double speed)> SeaStateParameters =
		new Dictionary<int, (double height, double speed)>
	{
		{0, (0.0, 0.0)},      // 风速(m/s), 波高(m)
        {1, (0.1, 1.5)},
		{2, (0.5, 3.0)},
		{3, (1.25, 5.0)},
		{4, (2.5, 7.5)},
		{5, (4.0, 10.0)},
		{6, (6.0, 12.5)},
		{7, (9.0, 15.0)},
		{8, (14.0, 18.0)},
		{9, (20.0, 25.0)}
	};

	public override double CalculateClutterReturn(电磁波 wave, AtmosphericCondition atmosphere) {
		// 使用改进的GIT模型(Georgia Institute of Technology)
		Vector3D directionToClutter = (Position - wave.Position).Normalize();
		double grazeAngle = Math.PI / 2 - Math.Acos(directionToClutter.DotProduct(new Vector3D(0, 0, 1)));

		// 计算标准化杂波系数
		double sigma0 = CalculateNormalizedClutterCoefficient(
			grazeAngle,
			wave.Frequency,
			wave.Polarization);

		// 考虑海况影响
		var (waveHeight, windSpd) = SeaStateParameters[SeaState];
		double seaStateEffect = Math.Pow(10, (SeaState - 3) * 0.5 / 10);

		// 考虑风向影响
		double windEffect = Math.Abs(Math.Cos(WindDirection));

		return sigma0 * seaStateEffect * windEffect;
	}

	private double CalculateNormalizedClutterCoefficient(
		double grazeAngle,
		double frequency,
		PolarizationType polarization) {
		// GIT模型参数
		double lambda = 3e8 / frequency;
		double k = 2 * Math.PI / lambda;

		// 基础杂波系数
		double sigma0 = -30; // dB

		// 考虑极化影响
		if (polarization == PolarizationType.Horizontal)
			sigma0 -= 5;

		// 掠射角依赖性
		sigma0 += 20 * Math.Log10(Math.Sin(grazeAngle));

		// 频率依赖性
		sigma0 += 10 * Math.Log10(k);

		return Math.Pow(10, sigma0 / 10);
	}

	public override double CalculateClutterDoppler(电磁波 wave) {
		// 考虑海浪运动引起的多普勒效应
		double waveVelocity = Math.Sqrt(9.81 * WaveHeight); // 使用重力波色散关系
		Vector3D waveMotion = new Vector3D(
			waveVelocity * Math.Cos(WindDirection),
			waveVelocity * Math.Sin(WindDirection),
			0
		);

		return wave.CalculateDopplerShift(Position, waveMotion);
	}
}