def set_motors(left, right, node):
    aw(node.set_variables(motors(left, right)))

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }