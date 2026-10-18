from typing import Any


class SharedConf:
    """
    Represents a hierarchical namespace based configuration. One can push names and pop names.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """
        Creates a new hierarchical configuration.
        :param data: The dictionary based config to use.
        """
        self.__data = data
        self.names: list[str] = []

        # overwrite build in functions to add annotations

    def __contains__(self, key: str) -> bool:
        """
        Check if key is contained using correct prefix..
        :param key: The key relative to the namespace.
        :return True if it is contained.
        """
        s = self.__to_prefix_identifier(self.names)
        return f"{s}{key}" in self.__data

    def get_root(self, key: str) -> Any:
        """
        Get item in configuration using correct prefix and key.
        :param key: The key relative to the namespace.
        :param default The default values.
        :return The extracted item or default value if not available.
        """
        return self.__data[f"{key}"]

    def get_namespace(self) -> str:
        names = self.names
        if len(names) > 0:
            return f"{'_'.join(names)}"
        else:
            return ""

    def get(self, key: str, default: Any) -> Any:
        """
        Get item in configuration using correct prefix and key.
        :param key: The key relative to the namespace.
        :param default The default values.
        :return The extracted item or default value if not available.
        """
        s = f"{self.__to_prefix_identifier(self.names)}"
        cont = f"{s}{key}" in self.__data
        if not cont:
            self.__data[f"{s}{key}"] = default
        return self.__data[f"{s}{key}"]

    def __getitem__(self, key: str) -> Any:
        """
        Get item in configuration using correct prefix and key.
        :param key: The key relative to the namespace.
        :return The extracted item.
        """
        s = f"{self.__to_prefix_identifier(self.names)}"
        return self.__data[f"{s}{key}"]

    def __setitem__(self, key: str, item: Any) -> None:
        """
        Set item in configuration using correct prefix and key to the passed item.
        :param key: The key relative to the namespace.
        :param item: The item to set.
        """
        s = self.__to_prefix_identifier(self.names)
        self.__data[f"{s}{key}"] = item

    def ns(self, name: str) -> "HConfWrapper":
        return HConfWrapper(self, name)

    def push(self, name: str) -> "SharedConf":
        """
        Push a new namespace internally.
        :param name: The new namespace to push.
        :return: The object itself.
        """
        self.names.append(name)
        return self

    def pop(self) -> "SharedConf":
        """
        Pops the last namespace, if one is available..
        :return: The object itself.
        """
        if len(self.names) > 0:
            self.names = self.names[:-1]
        return self

    def __to_prefix_identifier(self, names: list[str]) -> str:
        """
        Converts the passed list of names to a valid prefix.
        :param names: A list of names to use.
        :return: The prefix to use.
        """
        if len(names) > 0:
            return f"{'_'.join(names)}_"
        else:
            return ""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a dictionary.
        :return: The data dictionary.
        """
        return self.__data


class HConfWrapper:
    def __init__(self, hconf: SharedConf, name: str) -> None:
        self.name = [name]
        self.hconf = hconf

    def __enter__(self) -> None:
        [self.hconf.push(name) for name in self.name]

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        [self.hconf.pop() for name in self.name]

    def add(self, name: str) -> "HConfWrapper":
        self.name.append(name)
        return self
