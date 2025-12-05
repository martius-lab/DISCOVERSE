# YAML Task Factory - Documentation Index

Welcome to the YAML Task Factory documentation! This system allows you to define robotic manipulation tasks using YAML configuration files.

## 📚 Documentation Files

### Getting Started
- **[README.md](README.md)** - Complete documentation with examples, tutorials, and troubleshooting
- **[QUICKREF.md](QUICKREF.md)** - Quick reference cheat sheet for experienced users
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design diagrams

### Technical Details
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation overview and deliverables

## 🚀 Quick Links

### For New Users
1. Start with [README.md - Quick Start](README.md#quick-start)
2. Run the demo: `python examples/tasks_airbot_play/demo_yaml_factory.py`
3. Try an example: `python examples/tasks_airbot_play/run_yaml_task.py --config discoverse/configs/tasks/pick_place_block.yaml`

### For Developers
1. See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Review [Motion Primitives](README.md#motion-primitives) for extending functionality
3. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for complete feature list

### For Task Designers
1. Use [QUICKREF.md](QUICKREF.md) as a cheat sheet
2. Start with example configs in `discoverse/configs/tasks/`
3. Validate with: `python -m discoverse.task_factory.yaml_utils CONFIG.yaml --validate`

## 📁 File Structure

```
discoverse/task_factory/
├── __init__.py                    # Module exports
├── task_factory.py                # Main factory implementation
├── motion_primitives.py           # Motion primitive library
├── yaml_utils.py                  # Validation and utilities
├── README.md                      # Complete documentation ⭐
├── QUICKREF.md                    # Quick reference
├── ARCHITECTURE.md                # System architecture
├── IMPLEMENTATION_SUMMARY.md      # Implementation details
└── INDEX.md                       # This file

discoverse/configs/tasks/
├── cover_cup_yaml.yaml           # Complex example
└── pick_place_block.yaml         # Simple example

examples/tasks_airbot_play/
├── run_yaml_task.py              # Main execution script
└── demo_yaml_factory.py          # Demo and learning tool
```

## 🎯 Common Tasks

### Create a New Task
1. Copy `pick_place_block.yaml` as template
2. Modify objects, actions, and success conditions
3. Validate: `python -m discoverse.task_factory.yaml_utils your_task.yaml --validate`
4. Run: `python run_yaml_task.py --config your_task.yaml`

### Collect Training Data
```bash
python run_yaml_task.py --config task.yaml --auto --data_set_size 1000
```

### Debug a Task
```bash
# Run with visualization
python run_yaml_task.py --config task.yaml

# Summarize configuration
python -m discoverse.task_factory.yaml_utils task.yaml --summarize
```

## 🔧 Components Overview

| Component | Purpose | Location |
|-----------|---------|----------|
| YAMLTaskFactory | Parse YAML and generate tasks | `task_factory.py` |
| Motion Primitives | Reusable action building blocks | `motion_primitives.py` |
| Validator | Check YAML correctness | `yaml_utils.py` |
| Run Script | Execute tasks | `run_yaml_task.py` |
| Demo Script | Learn the system | `demo_yaml_factory.py` |
| Examples | Template configurations | `configs/tasks/*.yaml` |

## 📖 Topics by Interest

### I want to...

**...understand the system**
- Read: [ARCHITECTURE.md](ARCHITECTURE.md)
- Run: `python demo_yaml_factory.py`

**...create a new task**
- Guide: [README.md - Quick Start](README.md#quick-start)
- Template: `configs/tasks/pick_place_block.yaml`
- Reference: [QUICKREF.md](QUICKREF.md)

**...use motion primitives**
- Documentation: [README.md - Motion Primitives](README.md#motion-primitives)
- Cheat sheet: [QUICKREF.md - Motion Primitives](QUICKREF.md#motion-primitives-cheat-sheet)

**...define success conditions**
- Documentation: [README.md - Success Conditions](README.md#success-conditions)
- Examples: See YAML files in `configs/tasks/`

**...add randomization**
- Guide: [README.md - Domain Randomization](README.md#domain-randomization)
- Example: `cover_cup_yaml.yaml`

**...extend the system**
- Adding primitives: [README.md - Creating Custom Motion Primitives](README.md#creating-custom-motion-primitives)
- Architecture: [ARCHITECTURE.md - Extension Points](ARCHITECTURE.md#extension-points)

**...troubleshoot issues**
- Guide: [README.md - Troubleshooting](README.md#troubleshooting)
- Validation: Run with `--validate` flag

## 💡 Examples

### Minimal Configuration
See: `configs/tasks/pick_place_block.yaml`

### Complex Multi-Step Task
See: `configs/tasks/cover_cup_yaml.yaml`

### Python API Usage
```python
from discoverse.task_factory import load_task_from_yaml
TaskClass, cfg, factory = load_task_from_yaml("config.yaml")
```

## 🆘 Getting Help

1. Check [README.md - Troubleshooting](README.md#troubleshooting)
2. Run validator: `python -m discoverse.task_factory.yaml_utils CONFIG.yaml --validate`
3. Try demos: `python demo_yaml_factory.py`
4. Review examples in `configs/tasks/`

## 🎓 Learning Path

**Beginner:**
1. Read [README.md - Quick Start](README.md#quick-start)
2. Run `python demo_yaml_factory.py`
3. Try `pick_place_block.yaml`

**Intermediate:**
1. Study `cover_cup_yaml.yaml`
2. Create your own task
3. Use [QUICKREF.md](QUICKREF.md) as reference

**Advanced:**
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Create custom motion primitives
3. Extend the factory system

## 📝 Version

- Implementation Date: December 2025
- Python Version: 3.8+
- Dependencies: See `pyproject.toml`

## 🔗 Related Documentation

- DISCOVERSE Main: `README.md` (project root)
- Task Base Classes: `discoverse/task_base/`
- Robot Configs: `discoverse/robots_env/`
- Examples: `examples/tasks_airbot_play/`

---

**Start here:** [README.md](README.md) for complete documentation
**Quick help:** [QUICKREF.md](QUICKREF.md) for quick reference
