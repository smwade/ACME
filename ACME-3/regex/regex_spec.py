
def prob1():
  pattern_string = r"." # Edit this line
  pattern = re.compile(pattern_string)
  print bool(pattern.match("^{(!%.*_)}&"))

def prob2():
    pattern_string = r"^(Book|Mattress|Grocery) (store|supplier)" # Edit this line
    pattern = re.compile(pattern_string)
    strings_to_match = [ "Book store", "Book supplier", "Mattress store", \
    "Mattress supplier", "Grocery store", "Grocery supplier"]
    not_to_match = ["Book store sale", "Grocery Book store"]
    print all(pattern.match(string) for string in strings_to_match) and not all(pattern.match(string) for string in not_to_match)

def prob3():
    pattern_string = r"^[^d-w]$" # Edit this line
    pattern = re.compile(pattern_string)
    strings_to_match = ["a", "b", "c", "x", "y", "z"]
    uses_line_anchors = (pattern_string.startswith('^') and pattern_string.endswith('$'))
    solution_is_clever = (len(pattern_string) == 8)
    matches_list = all(pattern.match(string) for string in strings_to_match)
    print uses_line_anchors and solution_is_clever and matches_list

def prob4():
    identifier_pattern_string = r"^[^0-9]\w\w\w\w$" # Edit this line
    identifier_pattern = re.compile(identifier_pattern_string)
    valid = ["mouse", "HORSE", "_1234", "__x__", "while"]
    not_valid = ["3rats", "err*r", "sq(x)", "too_long"]
    print all(identifier_pattern.match(string) for string in valid) and not any(identifier_pattern.match(string) for string in not_valid)

def prob5():
    identifier_pattern_string = r"^[^0-9]\w+$" #Edit this line
    identifier_pattern = re.compile(identifier_pattern_string)
    valid = ["mouse", "HORSE", "_1234", "__x__", "while","Longer_String_Than_Before"]
    not_valid = ["3rats", "err*r", "sq(x)"]
    print all(identifier_pattern.match(string) for string in valid) and not any(identifier_pattern.match(string) for string in not_valid)

def prob6():
    date_pattern = re.compile(r".*[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}")
    phone_pattern = re.compile(r".*\(?[0-9]{3}\)?\-[0-9]{3}\-[0-9]{4}")
    email_pattern = re.compile(r".*\w+@\w+\.[a-zA-Z]{2,}")

    contact_dict = {}
    with open("contacts.txt") as inFile:
        for line in inFile:
            user = {}
            word_list = re.split("\s+", line)[:-1]
            name = " ".join(word_list[:2])
            for word in word_list[2:]:
                phone_result = phone_pattern.match(word)
                date_result = date_pattern.match(word)
                email_result = email_pattern.match(word)
                if phone_result:
                    user["phone"] = phone_result.group(0)
                if date_result:
                    user["bday"] = date_result.group(0)
                if email_result:
                    user["email"] = email_result.group(0)
            contact_dict[name] = user
    return contact_dict


def prob7():
    pass


