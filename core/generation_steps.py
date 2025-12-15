from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Set, Tuple

class StepStatus(Enum):
    """步骤状态枚举"""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass
class GenerationStep:
    """地图生成步骤的基类"""
    id: str
    name: str
    description: str
    # 步骤依赖项列表 - 这些步骤必须在此步骤之前完成
    dependencies: List[str] = field(default_factory=list)
    # 步骤提供的数据标识符列表 - 这些数据由此步骤生成
    provides: List[str] = field(default_factory=list)
    # 步骤所需的数据标识符列表 - 这些数据需要由其他步骤提供
    requires: List[str] = field(default_factory=list)
    # 步骤是否默认启用
    enabled: bool = True
    # 步骤是否可选
    optional: bool = False
    # 是否为高级步骤（高级步骤默认在UI中折叠）
    advanced: bool = False
    # 步骤的当前状态
    status: StepStatus = StepStatus.NOT_STARTED
    # 步骤执行函数（实际的生成函数）
    execute_func: Optional[Callable] = None
    # 步骤的UI配置（前端显示配置）
    ui_config: Dict[str, Any] = field(default_factory=dict)
    # 步骤参数配置
    params: Dict[str, Any] = field(default_factory=dict)

    def execute(self, context: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """执行此生成步骤，更新并返回上下文"""
        if logger:
            logger.log(f"执行步骤: {self.name}")
        self.status = StepStatus.IN_PROGRESS
        
        try:
            if self.execute_func:
                # 调用执行函数，传入上下文与参数
                result = self.execute_func(context, self.params, logger)
                # 更新上下文
                context.update(result)
                self.status = StepStatus.COMPLETED
            else:
                if logger:
                    logger.log(f"警告: 步骤 {self.name} 没有实现执行函数")
                self.status = StepStatus.SKIPPED
        except Exception as e:
            if logger:
                logger.log(f"步骤 {self.name} 执行失败: {str(e)}", "ERROR")
            self.status = StepStatus.FAILED
            raise
            
        return context
    
    def reset(self):
        """重置步骤状态"""
        self.status = StepStatus.NOT_STARTED

class StepManager:
    """管理地图生成步骤的类"""
    def __init__(self):
        self.steps: Dict[str, GenerationStep] = {}
        self.step_order: List[str] = []
        self._register_default_steps()
    
    def _register_default_steps(self):
        """注册默认的生成步骤"""
        # 这里将逐个注册所有默认步骤
        pass
    
    def register_step(self, step: GenerationStep):
        """注册一个生成步骤"""
        self.steps[step.id] = step
        if step.id not in self.step_order:
            self.step_order.append(step.id)
    
    def get_step(self, step_id: str) -> Optional[GenerationStep]:
        """获取指定ID的步骤"""
        return self.steps.get(step_id)
    
    def get_all_steps(self) -> List[GenerationStep]:
        """获取所有注册的步骤"""
        return [self.steps[step_id] for step_id in self.step_order if step_id in self.steps]
    
    def get_enabled_steps(self) -> List[GenerationStep]:
        """获取所有启用的步骤"""
        return [self.steps[step_id] for step_id in self.step_order 
                if step_id in self.steps and self.steps[step_id].enabled]
    
    def set_step_order(self, step_ids: List[str]):
        """设置步骤执行顺序"""
        # 验证所有ID都是有效的
        for step_id in step_ids:
            if step_id not in self.steps:
                raise ValueError(f"未知的步骤ID: {step_id}")
        self.step_order = step_ids
    
    def validate_dependencies(self) -> List[str]:
        """验证步骤依赖关系，返回检测到的错误"""
        errors = []
        enabled_steps = self.get_enabled_steps()
        enabled_step_ids = [step.id for step in enabled_steps]
        
        # 检查每个启用步骤的依赖项
        for step in enabled_steps:
            for dep_id in step.dependencies:
                # 依赖的步骤必须存在
                if dep_id not in self.steps:
                    errors.append(f"步骤 '{step.name}' 依赖于不存在的步骤 ID: {dep_id}")
                    continue
                    
                # 依赖的步骤必须启用
                if dep_id not in enabled_step_ids:
                    errors.append(f"步骤 '{step.name}' 依赖于未启用的步骤: {self.steps[dep_id].name}")
                    continue
                    
                # 依赖的步骤必须在当前步骤之前执行
                if self.step_order.index(dep_id) >= self.step_order.index(step.id):
                    errors.append(f"步骤 '{step.name}' 必须在其依赖项 '{self.steps[dep_id].name}' 之后执行")
        
        # 检查数据依赖
        provided_data = set()
        for step_id in self.step_order:
            step = self.steps[step_id]
            if not step.enabled:
                continue
                
            # 检查所需数据是否由前面的步骤提供
            for req in step.requires:
                if req not in provided_data:
                    errors.append(f"步骤 '{step.name}' 需要数据 '{req}'，但该数据未由任何前序步骤提供")
            
            # 添加此步骤提供的数据
            provided_data.update(step.provides)
                
        return errors
    
    def execute_steps(self, context: Dict[str, Any], logger=None) -> Dict[str, Any]:
        """按顺序执行所有启用的步骤"""
        # 验证依赖关系
        errors = self.validate_dependencies()
        if errors:
            error_msg = "\n".join(errors)
            if logger:
                logger.log(f"依赖关系验证失败:\n{error_msg}", "ERROR")
            raise ValueError(f"依赖关系验证失败:\n{error_msg}")
        
        # 重置所有步骤状态
        for step in self.steps.values():
            step.reset()
        
        # 按顺序执行步骤
        for step_id in self.step_order:
            step = self.steps[step_id]
            if not step.enabled:
                if logger:
                    logger.log(f"跳过禁用的步骤: {step.name}")
                step.status = StepStatus.SKIPPED
                continue
                
            try:
                if logger:
                    logger.log(f"开始执行步骤: {step.name}")
                context = step.execute(context, logger)
                if logger:
                    logger.log(f"步骤 {step.name} 执行完成")
            except Exception as e:
                if logger:
                    logger.log(f"步骤 {step.name} 执行失败: {str(e)}", "ERROR")
                # 如果是可选步骤，继续执行后面的步骤
                if step.optional:
                    if logger:
                        logger.log(f"由于 {step.name} 是可选步骤，继续执行")
                    continue
                # 否则向上抛出异常
                raise
        
        return context