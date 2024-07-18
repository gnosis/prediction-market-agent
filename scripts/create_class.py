import types


class BaseClass:
    def base_method(self):
        return "This is a method from the BaseClass"


class ClassFactory:
    def create_class(self, class_name, base_classes=(), attributes=None):
        if attributes is None:
            attributes = {}

        # Dynamically create the new class using the type function
        new_class = type(class_name, base_classes, attributes)
        return new_class


# Example usage
factory = ClassFactory()


def init2(self):
    self.a = 1


function_code = """
def dynamic_function(self, add_value: int = 0, append_str: str = '') -> str:
    return f"Dynamic function output: attr1={{self.attr1 + add_value}}, attr2={{self.attr2 + append_str}}"
"""

# Create a namespace dictionary to compile the function code
namespace = {}

# Execute the function code in the namespace
exec(function_code, namespace)

# Retrieve the dynamic function from the namespace
dynamic_function = namespace["dynamic_function"]


# Convert the dynamic function to a method using types.MethodType
dynamic_function2 = types.FunctionType(
    dynamic_function.__code__,
    dynamic_function.__globals__,
    name="dynamic_function",
    argdefs=(),
    closure=dynamic_function.__closure__,
)


# Define some attributes and methods for the dynamic class
attributes = {
    "attr1": 42,
    "method1": lambda self: f"Hello from {self.__class__.__name__}",
    "__init__": init2,
}

# 1 - local filesystem, eval(string), register on engine
# 2 - store string on IPFS, download it, eval(string_, register on engine (benefit - everyone can use it,
# all other agents)
# 3 - db instead of IPFS

# Create a dynamic class named 'DynamicClass'
DynamicClass = factory.create_class("DynamicClass", (), attributes)

# Instantiate the dynamic class
dynamic_instance = DynamicClass()

# Access its attributes and methods
print(dynamic_instance.attr1)  # Output: 42
print(dynamic_instance.method1())
print(dynamic_instance.a)
