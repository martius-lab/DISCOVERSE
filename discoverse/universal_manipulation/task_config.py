"""
任务配置加载器

用于加载和解析任务配置文件，提供统一的任务定义接口。
"""

import os
import yaml
import re
from typing import Dict, List, Any, Optional

from .config_utils import load_and_resolve_config, replace_variables
from .predicates import build_metadata_map, ObjectMetadata

class TaskConfigLoader:
    """任务配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化任务配置加载器
        
        Args:
            config_path: 任务配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.runtime_params = {}
        
        if config_path:
            self.load_config(config_path)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfigLoader':
        """从字典创建配置加载器
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置加载器实例
        """
        loader = cls()
        loader.config = config_dict
        loader._post_process_config()
        loader._validate_config()
        return loader
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载任务配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            任务配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        try:
            # 使用模板化配置解析
            self.config = load_and_resolve_config(config_path)
            self.config = replace_variables(self.config)

            self._post_process_config()
            self._validate_config()

            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse task config file {config_path}: {e}")
    
    def _post_process_config(self):
        """补充和缓存扩展字段"""
        self._objects_metadata: Dict[str, ObjectMetadata] = build_metadata_map(
            self.config.get("objects", [])
        )
        self._goal_expression: Optional[str] = self.config.get("goal")
        self._base_goal_family: Optional[str] = self.config.get("base_goal_family")
        self._obstructions: List[Any] = self.config.get("obstructions", [])
        self._expected_min_steps: Optional[int] = self.config.get("expected_min_steps")
        self._subgoals: List[Any] = self.config.get("subgoals", [])

        if self._goal_expression and not self.config.get("success_check"):
            self.config["success_check"] = self._compile_goal_to_success_check(
                self._goal_expression
            )

    def _validate_config(self):
        """验证任务配置文件的必要字段"""
        required_fields = [
            'task_name',
            'description'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in task config: {field}")
        
        # 检查状态字段（支持states或task_states）
        if 'states' not in self.config and 'task_states' not in self.config:
            raise ValueError("Task must have 'states' or 'task_states' field")
        
        # 验证状态配置（优先使用states，然后是task_states）
        states = self.config.get('states', self.config.get('task_states', []))
        if not isinstance(states, list) or len(states) == 0:
            raise ValueError("Task must have at least one state")
        
        for i, state in enumerate(states):
            if 'name' not in state:
                raise ValueError(f"State {i} missing required field: name")
            if 'primitive' not in state:
                raise ValueError(f"State {i} ({state['name']}) missing required field: primitive")
    
    def resolve_parameters(self, value: Any) -> Any:
        """
        解析参数化的值，支持 {param} 和 ${param} 格式
        
        Args:
            value: 要解析的值
            
        Returns:
            解析后的值
        """
        if isinstance(value, str):
            # 处理 ${param} 格式的参数替换
            import re
            
            def replace_param(match):
                param_name = match.group(1)
                # 优先使用运行时参数
                if param_name in self.runtime_params:
                    return str(self.runtime_params[param_name])
                # 然后使用配置文件中的运行时参数
                elif 'runtime_parameters' in self.config and param_name in self.config['runtime_parameters']:
                    return str(self.config['runtime_parameters'][param_name])
                # 最后使用旧格式的parameters
                elif param_name in self.config.get('parameters', {}):
                    return str(self.config['parameters'][param_name])
                # 如果找不到参数，保持原样
                return match.group(0)
            
            # 替换 ${param} 格式
            value = re.sub(r'\$\{([^}]+)\}', replace_param, value)
            # 替换 {param} 格式
            value = re.sub(r'\{([^}]+)\}', replace_param, value)
            
        elif isinstance(value, dict):
            return {k: self.resolve_parameters(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_parameters(item) for item in value]
        
        return value
    
    def get_resolved_states(self) -> List[Dict[str, Any]]:
        """
        获取解析后的状态序列
        
        Returns:
            解析后的状态列表
        """
        resolved_states = []
        
        # 获取状态列表（优先使用states，然后是task_states）
        states = self.config.get('states', self.config.get('task_states', []))
        
        for state in states:
            resolved_state = self.resolve_parameters(state.copy())
            resolved_states.append(resolved_state)
        
        return resolved_states
    
    # ============== 属性访问方法 ==============
    @property
    def task_name(self) -> str:
        """获取任务名称"""
        return self.config.get('task_name', 'unknown_task')

    @property 
    def record_fps(self) -> int:
        """获取记录帧率"""
        return self.config.get('observation', {'fps': 30}).get('fps', 30)
    
    @property
    def camera_configs(self) -> List[Dict[str, Any]]:
        """获取相机配置列表"""
        return self.config.get('observation', {}).get('cameras', [])

    @property 
    def success_check(self) -> Optional[Dict[str, Any]]:
        """获取成功检查配置"""
        return self.config.get('success_check')
    
    @property
    def goal_expression(self) -> Optional[str]:
        return self._goal_expression

    @property
    def base_goal_family(self) -> Optional[str]:
        return self._base_goal_family

    @property
    def obstructions(self) -> List[Any]:
        return self._obstructions

    @property
    def expected_min_steps(self) -> Optional[int]:
        return self._expected_min_steps

    @property
    def subgoals(self) -> List[Any]:
        return self._subgoals

    @property
    def objects_metadata(self) -> Dict[str, ObjectMetadata]:
        return self._objects_metadata
    
    @property
    def randomization(self) -> Optional[Dict[str, Any]]:
        """获取随机化配置"""
        return self.config.get('randomization')
    
    # ============== 随机化相关方法 ==============
    def validate_randomization_config(self) -> bool:
        """
        验证随机化配置的有效性
        
        Returns:
            是否有效
        """
        # 检查物体随机化配置
        if 'objects' in self.randomization:
            for i, obj_config in enumerate(self.randomization['objects']):
                if isinstance(obj_config, dict):
                    if 'name' not in obj_config:
                        print(f"❌ 随机化物体配置 {i} 缺少 'name' 字段")
                        return False
        return True

    # ------------------------------------------------------------------
    # Goal compilation helpers

    def _compile_goal_to_success_check(self, goal_expr: str) -> Dict[str, Any]:
        """
        将goal表达式解析为success_check配置。
        当前实现支持单层的 AND / OR 表达式。
        """

        clean_expr = goal_expr.strip()
        if "∨" in clean_expr or re.search(r"\bor\b", clean_expr, flags=re.IGNORECASE):
            operator = "or"
            parts = re.split(r"\s*(?:∨|\bor\b)\s*", clean_expr, flags=re.IGNORECASE)
        else:
            operator = "and"
            parts = re.split(r"\s*(?:∧|\band\b)\s*", clean_expr, flags=re.IGNORECASE)

        conditions = []
        for raw in parts:
            raw = raw.strip()
            if raw.startswith("(") and raw.endswith(")"):
                raw = raw[1:-1].strip()
            if not raw:
                continue
            match = re.match(r"([A-Za-z_]+)\((.*)\)", raw)
            if not match:
                continue
            predicate = match.group(1).lower()
            args = [arg.strip() for arg in match.group(2).split(",")]

            condition: Dict[str, Any] = {}
            if predicate == "on" and len(args) >= 2:
                condition = {"type": "on", "object": args[0], "support": args[1]}
            elif predicate == "in" and len(args) >= 2:
                condition = {"type": "in", "object": args[0], "container": args[1]}
            elif predicate == "upright" and len(args) >= 1:
                threshold = float(args[1]) if len(args) > 1 else 0.95
                condition = {
                    "type": "upright",
                    "object": args[0],
                    "threshold": threshold,
                }
            elif predicate == "at" and len(args) >= 2:
                region = args[1]
                tolerance = float(args[2]) if len(args) > 2 else 0.05
                condition = {
                    "type": "at",
                    "object": args[0],
                    "region": region,
                    "tolerance": tolerance,
                }
            elif predicate == "held" and len(args) >= 1:
                condition = {"type": "held", "object": args[0]}
            elif predicate == "inserted" and len(args) >= 2:
                depth = float(args[2]) if len(args) > 2 else 0.02
                angle_tol = float(args[3]) if len(args) > 3 else 0.1
                condition = {
                    "type": "inserted",
                    "peg": args[0],
                    "hole": args[1],
                    "depth": depth,
                    "angle_tol": angle_tol,
                }
            elif predicate == "clear" and len(args) >= 1:
                condition = {"type": "clear", "support": args[0]}
            elif predicate == "access" and len(args) >= 1:
                condition = {"type": "access", "object": args[0]}
            elif predicate == "open" and len(args) >= 1:
                condition = {"type": "open", "object": args[0]}
            elif predicate == "closed" and len(args) >= 1:
                condition = {"type": "closed", "object": args[0]}
            elif predicate == "visible" and len(args) >= 1:
                condition = {"type": "visible", "object": args[0]}

            if condition:
                conditions.append(condition)

        return {
            "method": "combined" if operator in ("and", "or") else "simple",
            "operator": operator,
            "conditions": conditions,
        }


    def __str__(self) -> str:
        """字符串表示"""
        if not self.config:
            return "TaskConfigLoader(not loaded)"
        
        return f"TaskConfigLoader({self.task_name}, {len(self.states)} states)"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def load_task_config(config_path: str) -> TaskConfigLoader:
    """
    便利函数：加载任务配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        任务配置加载器实例
    """
    return TaskConfigLoader(config_path) 
