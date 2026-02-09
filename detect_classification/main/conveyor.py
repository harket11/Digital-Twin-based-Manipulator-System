import numpy as np
import sys
import carb
import random

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicSphere

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

import omni
import omni.graph.core as og
import time
from pxr import Usd, UsdGeom, Gf
A_USD_PATH = ""
B_USD_PATH = "/home/rokey/Downloads/SimReady_Furniture_Misc_01_NVD@10010/Assets/simready_content/common_assets/props/pomegranate01/pomegranate01.usd"
C_USD_PATH = "/home/rokey/Downloads/SimReady_Furniture_Misc_01_NVD@10010/Assets/simready_content/common_assets/props/lychee01/lychee01.usd"
DROP_ROOT = "/World/AppleDrop"

TOTAL_APPLES = 30
i=0
SPAWN_EVERY_N_FRAMES = 100 #사과 드롭 주기

# UR10 근처
DROP_CENTER_X, DROP_CENTER_Y = 0, -5.5
SPREAD_XY = 0.25
START_Z = 1.8
HEIGHT_JITTER = 0.6

# 물리
COLLIDER_RADIUS = 0.07
DENSITY = 250.0
ENABLE_CCD = True

# 비주얼
VIS_SCALE = 1.5
VIS_SCALE2 = VIS_SCALE*2.1
VIS_Z_OFFSET = -0.075

HIDE_COLLIDER_VISUALLY = True

MIN_SEPARATION = COLLIDER_RADIUS * 2.2

# USD helpers

# 씬 전체(stage) 가져오기
def get_stage_safely():
    return omni.usd.get_context().get_stage()

# 경로에 Xform prim 없으면 만들기
def ensure_xform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    return prim

# 기존 변환은 유지하고 위치만 바꾸기
def set_translate_only(prim, x, y, z):
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(x, y, z))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

# 기존 변환은 유지하고 스케일만 바꾸기
def set_scale_uniform(prim, s):
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(s, s, s))
            return
    xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))

# 하위 prim들이 숨겨져 있으면 다시 보이게 풀어주기
def force_visible_recursive(root_prim):
    if not root_prim or not root_prim.IsValid():
        return
    for p in Usd.PrimRange(root_prim):
        img = UsdGeom.Imageable(p)
        if img:
            img.GetVisibilityAttr().Set("inherited")

# RMPFlow Controller
class RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:

        if attach_gripper:
            cfg = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflowSuction")
        else:
            cfg = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**cfg)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        (self._default_position, self._default_orientation,) = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation
        )

# Main
class Conveyor(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.target_position = np.array([0.0, 0.89, 0.75])

        self.task_phase = -1
        self._wait_counter = 0

        self.pick_position = np.array([0.0, 0.89, 0.75])
        self.robot_position = np.array([0.6, 0.5, 0.5])
        self.place_position_A = np.array([1, 0.5, 0.75])
        self.place_position_C = np.array([1, 1.0, 0.75])

        self._stage = None
        self._apple_i = 0
        self._apple_wait = 0

        self._spawned_xy = []
        self.apples = []
        self.select_rail=0
        self.rail1_i=0
        # Action Graph에서 만든 "등급 Subscriber 노드" 경로
        self._grade_sub_path = "/World/Background/subscriber"

        self._last_grade = None
        self.current_grade = None   # 사과 등급 (A/B/C)
        self._run_fsm = False
        self._armed_time = None
        self._trigger_delay = 0.0
        self._grade_sub_paths = ["/World/Background/subscriber"]

    def setup_scene(self):
        world = self.get_world()

        self.background_usd = "/home/rokey/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/hello_world/back.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")

        world.scene.add_default_ground_plane()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            sys.exit()

        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper"
        )

        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10",
                name="my_ur10",
                end_effector_prim_path="/World/UR10/ee_link",
                gripper=gripper
            )
        )

        ur10.set_joints_default_state(
            positions=np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
        )

    async def setup_post_load(self):
        self._world = self.get_world()

        self._stage = get_stage_safely()
        if self._stage is None:
            carb.log_error("USD stage is None in setup_post_load()")
            return

        ensure_xform(self._stage, "/World")
        ensure_xform(self._stage, DROP_ROOT)

        self.robots = self._world.scene.get_object("my_ur10")

        self.cspace_controller = RMPFlowController(
            name="my_ur10_cspace_controller",
            robot_articulation=self.robots,
            attach_gripper=True
        )

        self.robots.set_world_pose(position=self.robot_position)

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()

        self.task_phase = -1
        self._wait_counter = 0
        self._apple_i = 0
        self._apple_wait = 0
        self._spawned_xy = []
        self.apples = []
        self.rail1=[]

    def _pick_spawn_xy(self):
        for _ in range(TOTAL_APPLES):
            px = random.uniform(DROP_CENTER_X - SPREAD_XY, DROP_CENTER_X + SPREAD_XY)
            py = random.uniform(DROP_CENTER_Y - SPREAD_XY, DROP_CENTER_Y + SPREAD_XY)
            ok = True
            for (x, y) in self._spawned_xy:
                dx = px - x
                dy = py - y
                if (dx*dx + dy*dy) ** 0.5 < MIN_SEPARATION:
                    ok = False
                    break
            if ok:
                self._spawned_xy.append((px, py))
                return px, py

        px = random.uniform(DROP_CENTER_X - SPREAD_XY, DROP_CENTER_X + SPREAD_XY)
        py = random.uniform(DROP_CENTER_Y - SPREAD_XY, DROP_CENTER_Y + SPREAD_XY)
        self._spawned_xy.append((px, py))
        return px, py
    def spawn_apple_dynamic(self, i: int):
        prim_path = f"{DROP_ROOT}/Apple_{i:03d}"
        if self._world.scene.get_object(f"apple_{i:03d}") is not None:
            return

        px, py = self._pick_spawn_xy()
        pz = START_Z + random.uniform(0.0, HEIGHT_JITTER)
        usd_list = [A_USD_PATH, B_USD_PATH, C_USD_PATH]
        labels   = [1, 2, 3]  # A=1, B=2, C=3
        weights  = [0.2, 0.6, 0.2]

        # idx = random.choices(labels, weights=weights, k=1)[0]     # 1/2/3
        # 재현성을 위한 수동 시드 설정
        seed = [3,1,2,2,3,2,2,2,2,1,3,1,2,1,2,2,2,3,2,1,2,3,2,2,2,1,2,1,2,3]
        idx = seed[i]     
        good_apple = usd_list[idx - 1]   # 선택된 USD 경로                      
        if idx==3:
            Scale=VIS_SCALE2
            apple_c=np.array([0.45,0.3,0.15])
        elif idx==2:
            Scale=VIS_SCALE
            apple_c=np.array([1.0,0.2,0.2])
        elif idx==1:
            Scale=VIS_SCALE
            apple_c=np.array([0.93,0.06,0.06])

        apple = self._world.scene.add(
            DynamicSphere(
                prim_path=prim_path,
                name=f"apple_{i:03d}",
                position=np.array([px, py, pz]),
                radius=COLLIDER_RADIUS,
                color=apple_c,
                mass=0.2,
            )
        )

        # 스폰 직후는 '중력만' 자연스럽게 먹도록 XY=0으로만 초기화
        vel = apple.get_linear_velocity()
        apple.set_linear_velocity(np.array([0.0, 0.0, vel[2]]))

        # 비주얼 스킨
        stage = self._stage
        vis_path = prim_path + "/Visual"
        vis_prim = UsdGeom.Xform.Define(stage, vis_path).GetPrim()
        vis_prim.GetReferences().AddReference(good_apple)

        set_translate_only(vis_prim, 0.0, 0.0, VIS_Z_OFFSET)
        set_scale_uniform(vis_prim, Scale)
        force_visible_recursive(vis_prim)

        self.select_rail = (self.select_rail % 3) + 1

        self.apples.append([apple, self.select_rail])

        if self.select_rail == 1:
            self.rail1.append(apple)
        i+=1
    def _apply_apple_phase_velocity(self):
        for apple in self.apples:
            pos, _ = apple[0].get_world_pose()
            vel = apple[0].get_linear_velocity()
            new_vel=vel
            if 0.6 < pos[2] < 1.2:
                if apple[1] == 1:
                    if pos[0]>0 :
                        x_pos=np.array([-0.05])
                    else :
                        x_pos=np.array([0.05])
                    new_vel = np.array([x_pos[0], 0.6, vel[2]])
                elif apple[1] == 2:
                    new_vel = np.array([0.3 * vel[0] + 0.2, 0.3, vel[2]])
                elif apple[1] == 3:
                    new_vel = np.array([0.3 * vel[0] - 0.2, 0.3, vel[2]])
            elif 0.6>=pos[2] :
                new_vel=np.array([0.8*vel[0],0.8*vel[1],vel[2]])
            apple[0].set_linear_velocity(new_vel)
        for apple in self.rail1 :
            pos, _ = apple.get_world_pose()
            if pos[1]>0.85 and pos[1]<0.9 :
                apple.set_linear_velocity(np.array([0.0, 0.0, 0.0]))

    def subscribe_grade_value(self):
        try:
            a = og.Controller.attribute("/World/Background/subscriber/ros2_subscriber.outputs:data")
            v = a.get()
            if v is None:
                return None

            # bytes 처리
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")

            # dict 형태면 {"data": "A"} 같은 케이스
            if isinstance(v, dict) and "data" in v:
                v = v["data"]

            # 객체 형태면 v.data 같은 케이스
            if hasattr(v, "data"):
                v = v.data

            s = str(v).strip()

            # 혹시 "A\n" / "data: A" 같은 이상한 포맷 대비
            if s.startswith("data:"):
                s = s.split("data:", 1)[1].strip()

            if s in ("A", "B", "C"):
                return s
            return None
        except Exception as e:
            print("[DBG] read fail:", e)
            return None
    def physics_step(self, step_size):
        now = time.time()

        # 0) 등급 구독 + A/C만 트리거
        grade = self.subscribe_grade_value()

        # 1초에 한 번만 상태 출력 (스팸 방지)
        if not hasattr(self, "_dbg_t"):
            self._dbg_t = 0.0
        if now - self._dbg_t > 1.0:
            self._dbg_t = now
            print(f"[DBG] grade={grade} run_fsm={self._run_fsm} armed={self._armed_time} phase={self.task_phase}")

        # A 또는 C만 트리거 예약 (동작 중이면 무시)
        if (not self._run_fsm) and (grade in ("A", "C")) and (grade is not None):
            if grade != self._last_grade:
                self._last_grade = grade
                self.current_grade = grade
                self._armed_time = now + self._trigger_delay
                print(f"[GRADE] got {grade} -> start in {self._trigger_delay}s")

        # 예약 시간이 지나면 FSM 시작
        if (not self._run_fsm) and (self._armed_time is not None) and (now >= self._armed_time):
            self._run_fsm = True
            self._armed_time = None
            self.task_phase = -1
            print(f"[FSM] START grade={self.current_grade}")

        # 1) 사과 속도 규칙은 항상 적용
        self._apply_apple_phase_velocity()

        # 2) 사과 스폰은 계속 진행
        if self._apple_i < TOTAL_APPLES:
            self._apple_wait += 1
            if self._apple_wait >= SPAWN_EVERY_N_FRAMES:
                self._apple_wait = 0
                self.spawn_apple_dynamic(self._apple_i)
                self._apple_i += 1

        # 3) 로봇 FSM은 트리거가 켜졌을 때만 진행
        if not self._run_fsm:
            return  # 등급 토픽 대기

        # 트리거 후 첫 진입이면 phase 1로 세팅
        if self.task_phase == -1:
            ensure_xform(self._stage, DROP_ROOT)
            self._wait_counter = 0
            self.task_phase = 1

        # ur10 조작
        if self.task_phase == 1:
            cube_position = self.pick_position
            if cube_position[0] >= -0.09:
                self.task_phase = 2

        elif self.task_phase == 2:
            if self._wait_counter < 1:
                self._wait_counter += 1
            else:
                self.target_position = self.pick_position
                self.task_phase = 3

        elif self.task_phase == 3:
            _target_position = self.target_position.copy() - self.robot_position
            _target_position[2] = 0.55
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 4

        elif self.task_phase == 4:
            _target_position = self.target_position.copy() - self.robot_position
            _target_position[2] = 0.32
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 5

        elif self.task_phase == 5:
            self.robots.gripper.close()
            self.task_phase = 6

        elif self.task_phase == 6:
            _target_position = self.target_position.copy() - self.robot_position
            _target_position[2] = 0.65
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 7

        elif self.task_phase == 7:
            # 등급에 따라 place 목표 선택
            g = getattr(self, "current_grade", None)
            if g == "C":
                target_place = self.place_position_C
            else:
                # A/B는 일단 A 위치로
                target_place = self.place_position_A

            _target_position = target_place - self.robot_position
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 8

        elif self.task_phase == 8:
            self.robots.gripper.open()
            self.task_phase = 9

        elif self.task_phase == 9:
            # place 후 안전 높이로
            g = getattr(self, "current_grade", None)
            if g == "C":
                target_place = self.place_position_C
            else:
                target_place = self.place_position_A

            _target_position = target_place - self.robot_position
            _target_position[2] = 0.5
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()

                # 루프
                self.task_phase = -1
                self._run_fsm = False
                self._armed_time = None

                self._last_grade = None

                print("[FSM] DONE -> waiting next grade")
