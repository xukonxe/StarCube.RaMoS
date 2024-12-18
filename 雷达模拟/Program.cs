// 主程序
using StarCube.RaMoS;

var environment = new 仿真环境(timeStep: 1e-6);
Console.WriteLine("\n=== Initial Setup ===");

// 设置目标位置和波束方向
Vector3D targetPos = new Vector3D(200, 0, 100);
Vector3D initialPos = new Vector3D(0, 0, 0);

// 计算初始波束方向直接指向目标
Vector3D initialDirection = (targetPos - initialPos).Normalize();
Debug.LogVector("Initial direction to target", initialDirection);

// 创建电磁波 - 确保初始方向正确指向目标
var wave = new 电磁波(
	frequency: 10e9,                // 10GHz
	amplitude: 1000.0,
	position: initialPos,
	direction: initialDirection,    // 使用计算出的目标方向
	beamwidthAz: Math.PI / 6,      // 30度
	beamwidthEl: Math.PI / 6,      // 30度
	maxGain: 30                    // 30dB最大增益
);
wave.InitialPower = 1000000; // 1MW
wave.MainLobeDirection = initialDirection; // 确保主瓣方向与传播方向一致

// 创建运动目标
var target = new MovingTarget(
	position: targetPos,
	velocity: new Vector3D(-50, 0, 0),
	acceleration: new Vector3D(0, 0, 0)
);
target.RCS = 10000;  // 100平方米

// 设置大气条件
var atmosphere = new AtmosphericCondition(
	temperature: 288.15,
	pressure: 1013.25,
	humidity: 20,
	rainRate: 0
);

// 添加到环境
environment.AddWave(wave);
environment.AddObject(target);
environment.UpdateAtmosphere(atmosphere);

// Debug initial setup
Debug.LogValue("Initial Power (W)", wave.InitialPower);
Debug.LogValue("Target RCS (m²)", target.RCS);
double initialAngle = Math.Acos(wave.Direction.DotProduct((target.Position - wave.Position).Normalize())) * 180 / Math.PI;
Debug.LogValue("Initial angle to target (degrees)", initialAngle);

// 运行仿真
for (int i = 0; i < 50; i++) {
	Console.WriteLine($"\n=== Step {i} ===");

	// 更新环境前的调试信息
	Debug.LogVector("Pre-update Target Position", target.Position);
	Debug.LogVector("Pre-update Wave Position", wave.Position);

	// 更新环境
	environment.Update();

	// 基本计算
	Vector3D directionToTarget = (target.Position - wave.Position).Normalize();
	double gain = wave.CalculateAntennaGain(directionToTarget);
	double distance = (target.Position - wave.Position).Magnitude();

	// 详细的天线增益计算调试
	DetailedDebug.LogAntennaGainCalculation(wave, directionToTarget);

	// 检查波束覆盖
	DetailedDebug.LogBeamCheck(wave, directionToTarget, gain);

	// 更新目标位置的调试
	DetailedDebug.LogTargetUpdate(target, environment.TimeStep);

	// 输出基本信息
	Console.WriteLine($"Target position: ({target.Position.X:F2}, {target.Position.Y:F2}, {target.Position.Z:F2})");
	Console.WriteLine($"Wave position: ({wave.Position.X:F2}, {wave.Position.Y:F2}, {wave.Position.Z:F2})");
	Console.WriteLine($"Distance: {distance:F2} m");
	Console.WriteLine($"Antenna gain: {gain:F2} dB");

	var objectsInBeam = environment.GetObjectsInBeam(wave);
	Console.WriteLine($"Objects in beam: {objectsInBeam.Count}");

	// 场强计算调试
	double fieldStrength = environment.CalculateTotalFieldStrength(target.Position);
	Debug.LogValue("Raw Field Strength", fieldStrength);
	fieldStrength *= 1e6;  // 单位转换

	var analysis = environment.AnalyzeMultiTargetScene();
	Console.WriteLine($"Field strength: {fieldStrength:E3}");
	Console.WriteLine($"Scattered power: {analysis.TotalScatteredPower:E3} W");

	if (analysis.TotalScatteredPower > 0) {
		Debug.LogValue("Non-zero scattered power detected", analysis.TotalScatteredPower);
	}

	// 如果目标在波束内但场强为0，输出警告
	if (objectsInBeam.Count > 0 && fieldStrength == 0) {
		Console.WriteLine("WARNING: Target in beam but field strength is 0!");
		Debug.LogValue("Initial Power", wave.InitialPower);
		Debug.LogValue("Distance", distance);
		Debug.LogValue("Gain", gain);
	}

	// 更新波和目标
	wave.Update(environment.TimeStep);
	target.Update(environment.TimeStep);
}