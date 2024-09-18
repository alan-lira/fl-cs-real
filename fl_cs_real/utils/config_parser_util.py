from configparser import ConfigParser
from pathlib import Path
from re import match, search, split
from typing import Type


def is_representation_of_none_type(value: str) -> bool:
    """
    Verifies if a string value represents a None type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a None type or not.
    """
    return not value or value == "None"


def cast_to_none() -> Type[None]:
    """
    Casts to None type.

    Returns:
        Type[None]: the cast value.
    """
    return None


def is_representation_of_bool_type(value: str) -> bool:
    """
    Verifies if a string value represents a bool type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a bool type or not.
    """
    return value in ["True", "Yes", "False", "No"]


def cast_to_bool(value: str) -> bool:
    """
    Casts to a bool type.

    Args:
        value (str): the string value under casting.

    Returns:
        bool: the cast value.
    """
    return value in ["True", "Yes"]


def is_representation_of_int_type(value: str) -> bool:
    """
    Verifies if a string value represents an int type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents an int type or not.
    """
    return bool(match(r"^\d+$", value))


def cast_to_int(value: str) -> int:
    """
    Casts to an int type.

    Args:
        value (str): the string value under casting.

    Returns:
        int: the cast value.
    """
    return int(value)


def is_representation_of_float_type(value: str) -> bool:
    """
    Verifies if a string value represents a float type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a float type or not.
    """
    return bool(match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def cast_to_float(value: str) -> float:
    """
    Casts to a float type.

    Args:
        value (str): the string value under casting.

    Returns:
        float: the cast value.
    """
    return float(value)


def is_representation_of_tuple_type(value: str) -> bool:
    """
    Verifies if a string value represents a tuple type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a bool type or not.
    """
    return value[0] == "(" and value[-1] == ")"


def cast_to_tuple_recursively(tuple_element_str: str,
                              tuple_pattern: str,
                              left_delimiter: str,
                              right_delimiter: str) -> tuple:
    """
    Casts to a tuple type, recursively.

    Args:
        tuple_element_str (str): the string representation of a tuple element.
        tuple_pattern (str): the tuple pattern to be used during the match searching.
        left_delimiter (str): the string representation of the tuple left delimiter.
        right_delimiter (str): the string representation of the tuple right delimiter.

    Returns:
        tuple: the cast value.
    """
    casted_tuple = ()
    left_delimiter_counter = tuple_element_str.count(left_delimiter)
    right_delimiter_counter = tuple_element_str.count(right_delimiter)
    if left_delimiter_counter == 1 and right_delimiter_counter == 1:
        inner_tuple = ()
        tuple_element_str = tuple_element_str.split(left_delimiter)[1].split(right_delimiter)[0]
        tuple_elements = tuple_element_str.split(",")
        for tuple_element in tuple_elements:
            tuple_element = tuple_element.strip()
            if is_representation_of_none_type(tuple_element):
                tuple_element = cast_to_none()
            elif is_representation_of_bool_type(tuple_element):
                tuple_element = cast_to_bool(tuple_element)
            elif is_representation_of_int_type(tuple_element):
                tuple_element = cast_to_int(tuple_element)
            elif is_representation_of_float_type(tuple_element):
                tuple_element = cast_to_float(tuple_element)
            elif is_representation_of_list_type(tuple_element):
                tuple_element = cast_to_list(tuple_element)
            elif is_representation_of_dict_type(tuple_element):
                tuple_element = cast_to_dict(tuple_element)
            if len(tuple_elements) == 1:
                return tuple_element
            inner_tuple = inner_tuple + (tuple_element,)
        return inner_tuple
    else:
        search_result = search(tuple_pattern, tuple_element_str)
        tuple_element_str = search_result.group(1)
        casted_tuple += (cast_to_tuple_recursively(tuple_element_str, tuple_pattern, left_delimiter, right_delimiter),)
    return casted_tuple


def cast_to_tuple(value: str) -> tuple:
    """
    Casts to a tuple type.

    Args:
        value (str): the string value under casting.

    Returns:
        tuple: the cast value.
    """
    resulting_tuple = ()
    if value:
        tuple_pattern = r"\((\(*.+?\)*)\)$"
        left_delimiter = "("
        right_delimiter = ")"
        tuple_element_pattern = r"([\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*),*\s*"
        if value[0] == left_delimiter and value[-1] == right_delimiter:
            value = value[1:-1]
        tuple_elements_str = [tuple_elem.strip() for tuple_elem in split(tuple_element_pattern, value) if tuple_elem]
        for index, _ in enumerate(tuple_elements_str):
            while tuple_elements_str[index].count(left_delimiter) != tuple_elements_str[index].count(right_delimiter):
                tuple_elements_str[index] = tuple_elements_str[index] + ", " + tuple_elements_str[index+1]
                del tuple_elements_str[index+1]
        for tuple_element in tuple_elements_str:
            tuple_element = left_delimiter + tuple_element + right_delimiter
            casted_tuple_element = cast_to_tuple_recursively(tuple_element,
                                                             tuple_pattern,
                                                             left_delimiter,
                                                             right_delimiter)
            resulting_tuple = resulting_tuple + (casted_tuple_element,)
    return resulting_tuple


def is_representation_of_list_type(value: str) -> bool:
    """
    Verifies if a string value represents a list type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a list type or not.
    """
    return value[0] == "[" and value[-1] == "]"


def cast_to_list_recursively(list_element_str: str,
                             list_pattern: str,
                             left_delimiter: str,
                             right_delimiter: str) -> list:
    """
    Casts to a list type, recursively.

    Args:
        list_element_str (str): the string representation of a list element.
        list_pattern (str): the list pattern to be used during the match searching.
        left_delimiter (str): the string representation of the list left delimiter.
        right_delimiter (str): the string representation of the list right delimiter.

    Returns:
        list: the cast value.
    """
    casted_list = []
    left_delimiter_counter = list_element_str.count(left_delimiter)
    right_delimiter_counter = list_element_str.count(right_delimiter)
    if left_delimiter_counter == 1 and right_delimiter_counter == 1:
        inner_list = []
        list_element_str = list_element_str.split(left_delimiter)[1].split(right_delimiter)[0]
        list_elements = list_element_str.split(",")
        for list_element in list_elements:
            list_element = list_element.strip()
            if is_representation_of_none_type(list_element):
                list_element = cast_to_none()
            elif is_representation_of_bool_type(list_element):
                list_element = cast_to_bool(list_element)
            elif is_representation_of_int_type(list_element):
                list_element = cast_to_int(list_element)
            elif is_representation_of_float_type(list_element):
                list_element = cast_to_float(list_element)
            elif is_representation_of_tuple_type(list_element):
                list_element = cast_to_tuple(list_element)
            elif is_representation_of_dict_type(list_element):
                list_element = cast_to_dict(list_element)
            if len(list_elements) == 1:
                return list_element
            inner_list.append(list_element)
        return inner_list
    else:
        search_result = search(list_pattern, list_element_str)
        list_element_str = search_result.group(1)
        casted_list.extend(cast_to_list_recursively(list_element_str, list_pattern, left_delimiter, right_delimiter))
    return casted_list


def cast_to_list(value: str) -> list:
    """
    Casts to a list type.

    Args:
        value (str): the string value under casting.

    Returns:
        list: the cast value.
    """
    resulting_list = []
    if value:
        list_pattern = r"\[(\[*.+?\]*)\]$"
        left_delimiter = "["
        right_delimiter = "]"
        list_element_pattern = r"([\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*),*\s*"
        if value[0] == left_delimiter and value[-1] == right_delimiter:
            value = value[1:-1]
        list_elements_str = [list_elem.strip() for list_elem in split(list_element_pattern, value) if list_elem]
        for index, _ in enumerate(list_elements_str):
            while list_elements_str[index].count(left_delimiter) != list_elements_str[index].count(right_delimiter):
                list_elements_str[index] = list_elements_str[index] + ", " + list_elements_str[index+1]
                del list_elements_str[index+1]
        for list_element_str in list_elements_str:
            list_element_str = left_delimiter + list_element_str + right_delimiter
            casted_list_element = cast_to_list_recursively(list_element_str,
                                                           list_pattern,
                                                           left_delimiter,
                                                           right_delimiter)
            resulting_list.append(casted_list_element)
    return resulting_list


def is_representation_of_dict_type(value: str) -> bool:
    """
    Verifies if a string value represents a dict type.

    Args:
        value (str): the string value to be verified.

    Returns:
        bool: whether the string value represents a dict type or not.
    """
    return value[0] == "{" and value[-1] == "}"


def cast_to_dict_recursively(dict_element_str: str,
                             dict_pattern: str,
                             left_delimiter: str,
                             right_delimiter: str) -> dict:
    """
    Casts to dict type, recursively.

    Args:
        dict_element_str (str): the string representation of a dict element.
        dict_pattern (str): the dict pattern to be used during the match searching.
        left_delimiter (str): the string representation of the dict left delimiter.
        right_delimiter (str): the string representation of the dict right delimiter.

    Returns:
        dict: the cast value.
    """
    casted_dict = {}
    left_delimiter_counter = dict_element_str.count(left_delimiter)
    right_delimiter_counter = dict_element_str.count(right_delimiter)
    if left_delimiter_counter == 1 and right_delimiter_counter == 1:
        inner_dict = {}
        dict_element_str = dict_element_str.split(left_delimiter)[1].split(right_delimiter)[0]
        dict_elements = [dict_elem.strip() for dict_elem in split(r",(?![^\[\]]*])", dict_element_str) if dict_elem]
        for dict_element in dict_elements:
            if dict_element:
                key = dict_element.partition(":")[0]
                value = dict_element.partition(":")[2].strip()
                if is_representation_of_none_type(value):
                    value = cast_to_none()
                elif is_representation_of_bool_type(value):
                    value = cast_to_bool(value)
                elif is_representation_of_int_type(value):
                    value = cast_to_int(value)
                elif is_representation_of_float_type(value):
                    value = cast_to_float(value)
                elif is_representation_of_tuple_type(value):
                    value = cast_to_tuple(value)
                elif is_representation_of_list_type(value):
                    value = cast_to_list(value)
                inner_dict.update({key: value})
        return inner_dict
    else:
        search_result = search(dict_pattern, dict_element_str)
        key = search_result.group(1)
        dict_element_str = search_result.group(2)
        casted_dict.update({key:
                           cast_to_dict_recursively(dict_element_str, dict_pattern, left_delimiter, right_delimiter)})
    return casted_dict


def cast_to_dict(value: str) -> dict:
    """
    Casts to dict type.

    Args:
        value (str): the string value under casting.

    Returns:
        dict: the cast value.
    """
    resulting_dict = {}
    if value:
        dict_pattern = r"\{(.+?)\s*:\s*(\{*.+?\}*)\}$"
        left_delimiter = "{"
        right_delimiter = "}"
        dict_element_pattern = r"([\{a-zA-Z0-9._\-@\\/\}:\s]*\s*:\s*[\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*)"
        if value[0] == left_delimiter and value[-1] == right_delimiter:
            value = value[1:-1]
        dict_elements_str = [dict_elem.strip() for dict_elem in split(dict_element_pattern, value) if dict_elem]
        for index, _ in enumerate(dict_elements_str):
            while dict_elements_str[index].count(left_delimiter) != dict_elements_str[index].count(right_delimiter):
                dict_elements_str[index] = dict_elements_str[index] + " " + dict_elements_str[index+1]
                del dict_elements_str[index+1]
        for dict_element_str in dict_elements_str:
            dict_element_str = left_delimiter + dict_element_str + right_delimiter
            casted_dict_element = cast_to_dict_recursively(dict_element_str,
                                                           dict_pattern,
                                                           left_delimiter,
                                                           right_delimiter)
            for casted_dict_key, casted_dict_value in casted_dict_element.items():
                if casted_dict_key in resulting_dict:
                    if isinstance(resulting_dict[casted_dict_key], dict):
                        resulting_dict[casted_dict_key].update(casted_dict_value)
                    elif isinstance(resulting_dict[casted_dict_key], list):
                        resulting_dict[casted_dict_key].append(casted_dict_value)
                else:
                    resulting_dict.update(casted_dict_element)
    return resulting_dict


def parse_config_section(config_file: Path,
                         section_name: str) -> dict:
    """
    Parses a config file's section.

    Args:
        config_file (Path): the config file.
        section_name (str): the section name to be parsed.

    Returns:
        dict: the parsed section.
    """
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    config_section_parsed = {}
    if cp.has_section(section_name):
        config_section = {key: value for key, value in cp[section_name].items()}
        for key, value in config_section.items():
            value = " ".join(value.splitlines()).replace("\'", "").replace("\"", "")
            if is_representation_of_none_type(value):
                config_section_parsed.update({key: cast_to_none()})
            elif is_representation_of_bool_type(value):
                config_section_parsed.update({key: cast_to_bool(value)})
            elif is_representation_of_int_type(value):
                config_section_parsed.update({key: cast_to_int(value)})
            elif is_representation_of_float_type(value):
                config_section_parsed.update({key: cast_to_float(value)})
            elif is_representation_of_tuple_type(value):
                config_section_parsed.update({key: cast_to_tuple(value)})
            elif is_representation_of_list_type(value):
                config_section_parsed.update({key: cast_to_list(value)})
            elif is_representation_of_dict_type(value):
                config_section_parsed.update({key: cast_to_dict(value)})
            else:
                config_section_parsed.update({key: value})
    return config_section_parsed


def get_option_value(config_file: Path,
                     section_name: str,
                     option_name: str) -> str:
    """
    Gets the value of an option that belongs to a config file's section.

    Args:
        config_file (Path): the config file.
        section_name (str): the section name.
        option_name (str): the option name from which the value will be retrieved.

    Returns:
        str: the option's value.
    """
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    option_value = cp.get(section=section_name, option=option_name)
    return option_value


def set_option_value(config_file: Path,
                     section_name: str,
                     option_name: str,
                     new_value: str) -> None:
    """
    Sets the value of an option that belongs to a config file's section.

    Args:
        config_file (Path): the config file.
        section_name (str): the section name.
        option_name (str): the option name whose value will be updated.
        new_value (str): the new value to set.

    Returns:
        None
    """
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    cp.set(section=section_name, option=option_name, value=new_value)
    with open(file=config_file, mode="w", encoding="utf-8") as cf:
        cp.write(cf)
