from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
import json
from pathlib import Path

@dataclass_json
@dataclass(frozen=True)
class Big:
    s1 : 'Small'
    s2 : 'Small'

@dataclass(frozen=True)
class Small:
    number : int
    first_name : str

b = Big(Small(3,"bruh"), Small(2,"brug"))


path = Path("./greg.json").absolute()
with open(path, "w") as outfile:
    outdict = asdict(b)
    outdict['version'] = 2
    json.dump(outdict, outfile, indent=4)

with open(path, "r") as infile:
    indict = json.load(infile)
    assert indict.pop('version', None) == 2
    a = Big.from_dict(indict)

assert b == a