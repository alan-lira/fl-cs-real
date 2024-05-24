from configparser import ConfigParser
from pathlib import Path
from re import match, search, split


def is_representation_of_none_type(value: str) -> bool:
    return not value or value == "None"


def cast_to_none() -> any:
    return None


def is_representation_of_bool_type(value: str) -> bool:
    return value in ["True", "Yes", "False", "No"]


def cast_to_bool(value: str) -> bool:
    return value in ["True", "Yes"]


def is_representation_of_int_type(value: str) -> bool:
    return bool(match(r"^\d+$", value))


def cast_to_int(value: str) -> int:
    return int(value)


def is_representation_of_float_type(value: str) -> bool:
    return bool(match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def cast_to_float(value: str) -> float:
    return float(value)


def is_representation_of_tuple_type(value: str) -> bool:
    return value[0] == "(" and value[-1] == ")"


def cast_to_tuple_recursively(tuple_element_str: str,
                              tuple_pattern: str,
                              left_delimiter: str,
                              right_delimiter: str) -> tuple:
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
    tuple_pattern = r"\((\(*.+?\)*)\)$"
    left_delimiter = "("
    right_delimiter = ")"
    tuple_element_pattern = r"([\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*),*\s*"
    tuple_elements_str = [tuple_elem.strip() for tuple_elem in split(tuple_element_pattern, value[1:-1]) if tuple_elem]
    for index, _ in enumerate(tuple_elements_str):
        while tuple_elements_str[index].count(left_delimiter) != tuple_elements_str[index].count(right_delimiter):
            tuple_elements_str[index] = tuple_elements_str[index] + ", " + tuple_elements_str[index+1]
            del tuple_elements_str[index+1]
    resulting_tuple = ()
    for tuple_element in tuple_elements_str:
        tuple_element = left_delimiter + tuple_element + right_delimiter
        casted_tuple_element = cast_to_tuple_recursively(tuple_element,
                                                         tuple_pattern,
                                                         left_delimiter,
                                                         right_delimiter)
        resulting_tuple = resulting_tuple + (casted_tuple_element,)
    return resulting_tuple


def is_representation_of_list_type(value: str) -> bool:
    return value[0] == "[" and value[-1] == "]"


def cast_to_list_recursively(list_element_str: str,
                             list_pattern: str,
                             left_delimiter: str,
                             right_delimiter: str) -> list:
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
    list_pattern = r"\[(\[*.+?\]*)\]$"
    left_delimiter = "["
    right_delimiter = "]"
    list_element_pattern = r"([\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*),*\s*"
    list_elements_str = [list_elem.strip() for list_elem in split(list_element_pattern, value[1:-1]) if list_elem]
    for index, _ in enumerate(list_elements_str):
        while list_elements_str[index].count(left_delimiter) != list_elements_str[index].count(right_delimiter):
            list_elements_str[index] = list_elements_str[index] + ", " + list_elements_str[index+1]
            del list_elements_str[index+1]
    resulting_list = []
    for list_element_str in list_elements_str:
        list_element_str = left_delimiter + list_element_str + right_delimiter
        casted_list_element = cast_to_list_recursively(list_element_str,
                                                       list_pattern,
                                                       left_delimiter,
                                                       right_delimiter)
        resulting_list.append(casted_list_element)
    return resulting_list


def is_representation_of_dict_type(value: str) -> bool:
    return value[0] == "{" and value[-1] == "}"


def cast_to_dict_recursively(dict_element_str: str,
                             dict_pattern: str,
                             left_delimiter: str,
                             right_delimiter: str) -> dict:
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
    dict_pattern = r"\{(.+?)\s*:\s*(\{*.+?\}*)\}$"
    left_delimiter = "{"
    right_delimiter = "}"
    dict_element_pattern = r"([\{a-zA-Z0-9._\-@\\/\}:\s]*\s*:\s*[\{\[\(a-zA-Z0-9:._\-@\\/\)\]\}\s]*)"
    dict_elements_str = [dict_elem.strip() for dict_elem in split(dict_element_pattern, value[1:-1]) if dict_elem]
    for index, _ in enumerate(dict_elements_str):
        while dict_elements_str[index].count(left_delimiter) != dict_elements_str[index].count(right_delimiter):
            dict_elements_str[index] = dict_elements_str[index] + " " + dict_elements_str[index+1]
            del dict_elements_str[index+1]
    resulting_dict = {}
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
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    config_section = {key: value for key, value in cp[section_name].items()}
    config_section_parsed = {}
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
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    option_value = cp.get(section=section_name, option=option_name)
    return option_value


def set_option_value(config_file: Path,
                     section_name: str,
                     option_name: str,
                     new_value: str) -> None:
    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=config_file, encoding="utf-8")
    cp.set(section=section_name, option=option_name, value=new_value)
    with open(file=config_file, mode="w", encoding="utf-8") as cf:
        cp.write(cf)
