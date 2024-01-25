Search.setIndex({"docnames": ["index", "notebooks/TEAM_13_geometry", "notebooks/createMesh", "notebooks/createMesh_gaps", "notebooks/createMesh_gaps_try", "notebooks/current_density", "notebooks/magnetostatics"], "filenames": ["index.rst", "notebooks\\TEAM_13_geometry.ipynb", "notebooks\\createMesh.ipynb", "notebooks\\createMesh_gaps.ipynb", "notebooks\\createMesh_gaps_try.ipynb", "notebooks\\current_density.ipynb", "notebooks\\magnetostatics.ipynb"], "titles": ["Welcome to FEM\u2019s documentation!", "Creating the 3D geometry for TEAM problem 13", "Geometry and mesh for motor", "Geometry and mesh for motor", "Geometry and mesh for motor", "Computing the current density j in the coil", "Magnetostatics solution"], "terms": {"creat": [0, 5], "3d": 0, "geometri": [0, 5], "problem": [0, 5], "3": [0, 1, 2, 3, 4, 5, 6], "8": [0, 2, 3, 4, 5], "see": 1, "thi": [1, 5], "link": 1, "http": 1, "www": 1, "compumag": 1, "org": 1, "wp": 1, "content": 1, "upload": 1, "2018": 1, "06": 1, "problem13": 1, "pdf": 1, "we": [1, 5], "gener": [1, 5], "mesh": [1, 5, 6], "us": [1, 2, 3, 4], "netgen": [1, 2, 3, 4, 5, 6], "s": [1, 2, 3, 4], "occ": [1, 2, 3, 4], "interfac": 1, "note": 1, "ngsolv": [1, 2, 3, 4], "isn": 1, "t": [1, 2, 3, 4, 5, 6], "need": [1, 5], "1": [1, 2, 3, 4, 5, 6], "import": [1, 2, 3, 4, 5, 6], "from": [1, 2, 3, 4, 5], "webgui": [1, 2, 3, 4], "draw": [1, 2, 3, 4], "drawgeo": [1, 2, 3, 4], "box1": 1, "box": 1, "pnt": [1, 5], "100": [1, 5], "50": [1, 2, 3, 4], "box2": 1, "75": [1, 2, 3, 4], "round": 1, "corner": 1, "corner1_ext": 1, "cyl1_ext": 1, "cylind": 1, "z": [1, 2, 3, 4, 5], "r": [1, 2, 3, 4, 5, 6], "25": [1, 2, 3, 4], "h": 1, "corner1_int": 1, "cyl1_int": 1, "corner2_ext": 1, "cyl2_ext": 1, "corner2_int": 1, "cyl2_int": 1, "corner3_ext": 1, "cyl3_ext": 1, "corner3_int": 1, "cyl3_int": 1, "corner4_ext": 1, "cyl4_ext": 1, "corner4_int": 1, "cyl4_int": 1, "ad": 1, "steel": 1, "part": 1, "coil_ful": 1, "mid_steel": 1, "6": [1, 2, 3, 4, 5, 6], "64": [1, 2, 3, 4], "2": [1, 2, 3, 4, 5, 6], "r_steel1": 1, "0": [1, 2, 3, 4, 5, 6], "5": [1, 2, 3, 4, 6], "15": [1, 3, 5], "10": [1, 2, 3, 4, 5], "120": 1, "65": [1, 2, 3, 4], "r_steel2": 1, "r_steel3": 1, "r_steel": 1, "l_steel1": 1, "l_steel2": 1, "l_steel3": 1, "l_steel": 1, "glue": [1, 2, 3, 4], "half_box_1": 1, "half_box_2": 1, "coil_half_box_1": 1, "coil_half_box_2": 1, "coil": [0, 1, 2, 3, 4], "ambient": 1, "200": 1, "full": 1, "fanci": 1, "color": [1, 2, 3, 4, 5, 6], "cuz": 1, "why": 1, "i": [1, 2, 3, 5], "got": 1, "bit": 1, "bore": 1, "face": [1, 2, 3, 4, 5, 6], "col": 1, "identif": [1, 2, 3, 4, 5], "name": [1, 2, 3, 4], "coil_fac": [1, 5], "r_steel_fac": 1, "l_steel_fac": 1, "mid_steel_fac": 1, "ambient_fac": [1, 6], "coil_cut_1": [1, 5], "12": [1, 2, 3, 4, 5], "coil_cut_2": 1, "mat": [1, 2, 3, 4], "geoocc": [1, 2, 3, 4], "occgeometri": [1, 2, 3, 4], "geooccmesh": [1, 2, 3, 4, 5, 6], "generatemesh": [1, 2, 3, 4], "clip": [1, 5], "dist": 1, "basewebguiscen": [1, 2, 4], "decid": [2, 3], "whether": [2, 3], "whole": [2, 3], "just": [2, 3], "an": [2, 3], "8th": [2, 3], "slice": [2, 3], "pizza_slic": [2, 3], "true": [2, 3, 4, 5, 6], "defin": [2, 3, 4, 5], "radii": [2, 3, 4], "point": [2, 3, 4, 5], "help": [2, 3, 4], "magnet": [2, 3, 4], "airgap": [2, 3, 4], "origin": [2, 3, 4], "inner": [2, 3, 4], "radiu": [2, 3, 4], "rotor": [2, 3, 4], "r1": [2, 3, 4, 5], "26": [2, 3, 4], "outer": [2, 3, 4], "r2": [2, 3, 4], "78": [2, 3, 4], "63225": [2, 3, 4], "slide": [2, 3, 4], "r4": [2, 3, 4], "8354999": [2, 3, 4], "stator": [2, 3, 4], "r6": [2, 3, 4], "79": [2, 3, 4], "03874999": [2, 3, 4], "r7": [2, 3, 4], "242": [2, 3, 4], "r8": [2, 3, 4], "116": [2, 3, 4], "magnet1": [2, 3, 4], "air": [2, 3, 4], "around": [2, 3, 4], "m1": [2, 3, 4], "69": [2, 3, 4], "23112999": [2, 3, 4], "7": [2, 3, 4, 5, 6], "535512": [2, 3, 4], "m2": [2, 3, 4], "74": [2, 3, 4], "828958945": [2, 3, 4], "830092744": [2, 3, 4], "m3": [2, 3, 4], "66": [2, 3, 4], "13621099700001": [2, 3, 4], "599935335": [2, 3, 4], "m4": [2, 3, 4], "60": [2, 3, 4], "53713": [2, 3, 4], "22": [2, 3, 4], "30748": [2, 3, 4], "a5": [2, 3, 4], "75636": [2, 3, 4], "749913": [2, 3, 4], "a6": [2, 3, 4], "06735": [2, 3, 4], "810523": [2, 3, 4], "a7": [2, 3, 4], "6868747": [2, 3, 4], "3184618": [2, 3, 4], "3506200": [2, 3, 4], "51379": [2, 3, 4], "a8": [2, 3, 4], "59": [2, 3, 4], "942145092": [2, 3, 4], "24": [2, 3, 4, 5], "083661604": [2, 3, 4], "magnet2": [2, 3, 4], "m5": [2, 3, 4], "58": [2, 3, 4, 5], "579985516": [2, 3, 4], "27": [2, 3, 4], "032444757": [2, 3, 4], "m6": [2, 3, 4], "867251151": [2, 3, 4], "28": [2, 3, 4], "663475405": [2, 3, 4], "m7": [2, 3, 4], "570096319": [2, 3, 4], "45": [2, 3, 4], "254032279": [2, 3, 4], "m8": [2, 3, 4], "54": [2, 3, 4], "282213127": [2, 3, 4], "43": [2, 3, 4], "625389857": [2, 3, 4], "a1": [2, 3, 4], "53": [2, 3, 4], "39099766": [2, 3, 4], "259392713": [2, 3, 4], "a2": [2, 3, 4], "55": [2, 3, 4], "775078884": [2, 3, 4], "386185578": [2, 3, 4], "a3": [2, 3, 4], "41521771": [2, 3, 4], "355776837": [2, 3, 4], "a4": [2, 3, 4], "12210917100001": [2, 3, 4], "707477175": [2, 3, 4], "nut": [2, 3, 4], "s1": [2, 3, 4], "04329892000": [2, 3, 4], "9538335974": [2, 3, 4], "s2": [2, 3, 4], "80": [2, 3, 4], "143057128": [2, 3, 4], "4": [2, 3, 4, 5, 6], "0037794254": [2, 3, 4], "s3": [2, 3, 4], "387321219": [2, 3, 4], "965459706": [2, 3, 4], "s4": [2, 3, 4], "98": [2, 3, 4], "78501315600001": [2, 3, 4], "9007973292": [2, 3, 4], "s5": [2, 3, 4], "44904989600001": [2, 3, 4], "9": [2, 3, 4, 5], "026606148400001": [2, 3, 4], "s6": [2, 3, 4], "086666706": [2, 3, 4], "5525611543": [2, 3, 4], "s7": [2, 3, 4], "980020247": [2, 3, 4], "4912415424": [2, 3, 4], "s8": [2, 3, 4], "88229587": [2, 3, 4], "4102654448": [2, 3, 4], "next": [2, 3, 4], "function": [2, 3, 4], "numpi": [2, 3, 4, 5, 6], "np": [2, 3, 4, 5], "sin": [2, 3, 4], "co": [2, 3, 4], "pi": [2, 3, 4], "def": [2, 3, 4, 5], "rotat": [2, 3, 4], "m": [2, 3, 4, 5, 6], "k": [2, 3, 4, 5, 6], "p": [2, 3, 4, 5, 6], "mx": [2, 3, 4], "my": [2, 3, 4], "return": [2, 3, 4], "drawmagnet1": [2, 3, 4], "m1new": [2, 3, 4], "m2new": [2, 3, 4], "m3new": [2, 3, 4], "m4new": [2, 3, 4], "a5new": [2, 3, 4], "a6new": [2, 3, 4], "a7new": [2, 3, 4], "a8new": [2, 3, 4], "seg1": [2, 3, 4], "segment": [2, 3, 4], "seg2": [2, 3, 4], "seg3": [2, 3, 4], "seg4": [2, 3, 4], "wire": [2, 3, 4], "air_seg1": [2, 3, 4], "air_seg2": [2, 3, 4], "air_seg3": [2, 3, 4], "air_seg4": [2, 3, 4], "air_magnet1_1": [2, 3, 4], "air_seg5": [2, 3, 4], "air_seg6": [2, 3, 4], "air_seg7": [2, 3, 4], "air_seg8": [2, 3, 4], "air_magnet1_2": [2, 3, 4], "drawmagnet2": [2, 3, 4], "m5new": [2, 3, 4], "m6new": [2, 3, 4], "m7new": [2, 3, 4], "m8new": [2, 3, 4], "a1new": [2, 3, 4], "a2new": [2, 3, 4], "a3new": [2, 3, 4], "a4new": [2, 3, 4], "air_magnet2_1": [2, 3, 4], "air_magnet2_2": [2, 3, 4], "drawstatornut": [2, 3, 4], "s1new": [2, 3, 4], "s2new": [2, 3, 4], "s3new": [2, 3, 4], "s4new": [2, 3, 4], "s5new": [2, 3, 4], "s6new": [2, 3, 4], "s7new": [2, 3, 4], "s8new": [2, 3, 4], "seg5": [2, 3, 4], "seg6": [2, 3, 4], "stator_coil": [2, 3, 4], "stator_air": [2, 3, 4], "air_gap_st": [2, 3, 4], "shape": [2, 3, 4, 5], "technolog": [2, 3, 4], "domain": [2, 3, 4], "h_max": [2, 3, 4], "h_air_gap": [2, 3, 4], "05": [2, 3, 4], "h_air_magnet": [2, 3, 4], "h_coil": [2, 3, 4], "h_stator_air": [2, 3, 4], "h_magnet": [2, 3, 4], "h_stator_iron": [2, 3, 4], "h_rotor_iron": [2, 3, 4], "h_shaft_iron": [2, 3, 4], "rotor_inn": [2, 3, 4], "circl": [2, 3, 4], "rotor_out": [2, 3, 4], "sliding_inn": [2, 3, 4], "sliding_out": [2, 3, 4], "stator_inn": [2, 3, 4], "stator_out": [2, 3, 4], "edg": [2, 3, 4], "rotor_iron": [2, 3, 4], "air_gap": [2, 3, 4], "air_gap_rotor": [2, 3, 4], "stator_iron": [2, 3, 4], "rang": [2, 3, 4], "48": [2, 3, 4], "str": [2, 3, 4], "append": [2, 3, 4, 5], "maxh": [2, 3, 4], "magnets_interfac": [2, 3, 4], "rotor_air": [2, 3, 4], "08": 2, "shaft_iron": [2, 3, 4], "geo": [2, 3, 4], "pizza": [2, 3, 4], "moveto": [2, 3, 4], "line": [2, 3, 4], "90": [2, 3, 4], "close": [2, 3, 4], "left": [2, 3, 4, 5], "32": [2, 5], "96": 2, "right": [2, 3, 4], "30": [2, 3, 4], "46": 2, "rot": [2, 3, 4], "axi": [2, 3, 4], "identifi": [2, 3, 4], "per": [2, 3, 4], "dsa": 2, "dim": [2, 3, 4, 5], "ng": [2, 3, 4], "ngsolvemesh": [2, 3, 4], "refin": [2, 3, 4], "secondord": [2, 3, 4], "plist": [2, 3, 4], "pair": [2, 3, 4], "ngmesh": [2, 3, 4], "getidentif": [2, 3, 4], "list": [2, 3, 4, 5], "vertic": [2, 3, 4], "drawmesh": [2, 3, 4], "object": [2, 3, 4, 5], "type": [2, 3, 4, 6], "posit": [2, 3, 4], "purpl": [2, 3, 4], "mperp_mag1": [2, 3, 4], "arrai": [2, 3, 4, 5], "507223091788922": [2, 3, 4], "861814791678634": [2, 3, 4], "mperp_mag2": [2, 3, 4], "250741225095427": [2, 3, 4], "968054150364350": [2, 3, 4], "mperp_mag3": [2, 3, 4], "968055971101187": [2, 3, 4], "250734195544481": [2, 3, 4], "mperp_mag4": [2, 3, 4], "861818474866413": [2, 3, 4], "507216833690415": [2, 3, 4], "mperp_mag5": [2, 3, 4], "mperp_mag6": [2, 3, 4], "mperp_mag7": [2, 3, 4], "mperp_mag8": [2, 3, 4], "mperp_mag9": [2, 3, 4], "mperp_mag10": [2, 3, 4], "mperp_mag11": [2, 3, 4], "mperp_mag12": [2, 3, 4], "mperp_mag13": [2, 3, 4], "mperp_mag14": [2, 3, 4], "mperp_mag15": [2, 3, 4], "mperp_mag16": [2, 3, 4], "mperp_mag": [2, 3, 4], "c_": [2, 3, 4, 5], "nu0": [2, 3, 4], "158095238095238": [2, 3, 4], "offset": [2, 3, 4], "polepair": [2, 3, 4], "gamma_correction_model": [2, 3, 4], "gamma": [2, 3, 4], "40": [2, 3, 4], "gamma_correction_timestep": [2, 3, 4], "phi0": [2, 3, 4], "180": [2, 3, 4], "f48": [2, 3, 4], "area_coils_uplu": [2, 3, 4], "r_": [2, 3, 4, 5], "area_coils_vminu": [2, 3, 4], "area_coils_wplu": [2, 3, 4], "area_coils_uminu": [2, 3, 4], "area_coils_vplu": [2, 3, 4], "area_coils_wminu": [2, 3, 4], "11": [2, 3, 4, 5], "i0peak": [2, 3, 4], "1555": [2, 3, 4], "63491861": [2, 3, 4], "phase_shift_i1": [2, 3, 4], "phase_shift_i2": [2, 3, 4], "phase_shift_i3": [2, 3, 4], "i1c": [2, 3, 4], "i2c": [2, 3, 4], "i3c": [2, 3, 4], "areaofonecoil": [2, 3, 4], "00018053718538758062": [2, 3, 4], "uplu": [2, 3, 4], "vminu": [2, 3, 4], "wplu": [2, 3, 4], "uminu": [2, 3, 4], "vplu": [2, 3, 4], "wminu": [2, 3, 4], "j3": [2, 3, 4], "zero": [2, 3, 4, 5], "sy": [2, 3, 4, 5], "path": [2, 3, 4, 5], "insert": [2, 3, 4, 5], "add": [2, 3, 4, 5], "parent": [2, 3, 4, 5], "directori": [2, 3, 4, 5], "pde": [2, 3, 4, 5, 6], "plotli": [2, 3, 4], "io": [2, 3, 4], "pio": [2, 3, 4], "render": [2, 3, 4], "default": [2, 3, 4], "notebook": [2, 3, 4], "ind_air_al": [2, 3, 4], "flatnonzero": [2, 3, 4], "core": [2, 3, 4], "defchararrai": [2, 3, 4], "find": [2, 3, 4, 5], "regions_2d": [2, 3, 4, 5], "ind_stator_rotor": [2, 3, 4], "iron": [2, 3, 4], "ind_magnet": [2, 3, 4], "ind_coil": [2, 3, 4], "ind_shaft": [2, 3, 4], "shaft": [2, 3, 4], "trig_air_al": [2, 3, 4], "where": [2, 3, 4, 5], "isin": [2, 3, 4], "trig_stator_rotor": [2, 3, 4], "trig_magnet": [2, 3, 4], "trig_coil": [2, 3, 4], "trig_shaft": [2, 3, 4], "vek": [2, 3, 4], "nt": [2, 3, 4, 5], "fig": [2, 3, 4], "pdemesh": [2, 3, 4], "pdesurf": [2, 3, 4, 5], "show": [2, 3, 4, 5, 6], "makeidentif": [2, 3], "femsol": 2, "c0": [2, 3, 4], "c1": [2, 3], "point0": [2, 3], "point1": [2, 3], "ind0": [2, 3], "argsort": [2, 3], "aa": [2, 3], "edges0": [2, 3], "edges1": [2, 3], "sort": [2, 3, 4], "edgecoord0": [2, 3], "dtype": [2, 3], "int": [2, 3, 5, 6], "edgecoord1": [2, 3], "all": [2, 3], "edgestovertic": [2, 3], "ident_point": [2, 3], "ident_edg": [2, 3], "print": [2, 3], "3360": 2, "8310": 2, "77": 2, "12245": 2, "8309": 2, "3363": 2, "3365": 2, "383": 2, "91": 2, "12273": 2, "8324": 2, "382": 2, "3781": 2, "8834": 2, "381": 2, "89": 2, "3779": 2, "3376": 2, "380": 2, "88": 2, "12272": 2, "3375": 2, "379": 2, "87": 2, "3776": 2, "3849": 2, "378": 2, "86": 2, "8771": 2, "3373": 2, "377": 2, "85": 2, "8770": 2, "3372": 2, "376": 2, "84": 2, "12279": 2, "8833": 2, "375": 2, "83": 2, "12270": 2, "8320": 2, "374": 2, "82": 2, "12269": 2, "8317": 2, "373": 2, "81": 2, "12278": 2, "8316": 2, "372": 2, "12251": 2, "8315": 2, "10024": 2, "10023": 2, "21": 2, "10472": 2, "10470": 2, "23": [2, 3], "10630": 2, "13078": 2, "38": 2, "11261": 2, "11372": 2, "1450": 2, "1434": 2, "11371": 2, "6961": 2, "1449": 2, "1433": 2, "6880": 2, "11363": 2, "1448": 2, "1432": 2, "6879": 2, "11405": 2, "1447": 2, "1431": 2, "11591": 2, "11362": 2, "1446": 2, "1430": 2, "7335": 2, "11361": 2, "1445": 2, "1429": 2, "13175": 2, "13174": 2, "1444": 2, "1428": 2, "6878": 2, "6867": 2, "1443": 2, "1427": 2, "6877": 2, "6866": 2, "1442": 2, "1426": 2, "11667": 2, "6865": 2, "76": 2, "584": 2, "574": 2, "587": 2, "573": 2, "2921": 2, "642": 2, "2925": 2, "644": 2, "2920": 2, "639": 2, "2915": 2, "640": 2, "2912": 2, "634": 2, "2911": 2, "631": 2, "2908": 2, "628": 2, "2910": 2, "627": 2, "2906": 2, "623": 2, "2902": 2, "624": 2, "2898": 2, "619": 2, "2900": 2, "618": 2, "2895": 2, "614": 2, "2894": 2, "613": 2, "2890": 2, "609": 2, "2891": 2, "612": 2, "2887": 2, "608": 2, "2886": 2, "607": 2, "2882": 2, "601": 2, "2881": 2, "600": [2, 5], "2876": 2, "596": 2, "2877": 2, "595": 2, "2872": 2, "591": 2, "2871": 2, "590": 2, "14": [2, 3, 5], "20": [2, 3, 4, 5], "16": [2, 3, 5], "154": 2, "149": 2, "156": 2, "150": 2, "164": 2, "159": 2, "165": 2, "161": 2, "270": 2, "171": 2, "272": 2, "170": 2, "11326": 2, "11251": 2, "11327": 2, "11249": 2, "11323": 2, "11244": 2, "11319": 2, "11246": 2, "11316": 2, "11241": 2, "11315": 2, "11242": 2, "11310": 2, "11237": 2, "11314": 2, "11236": 2, "11308": 2, "11231": 2, "11307": 2, "11230": 2, "11302": 2, "11226": 2, "11304": 2, "11227": 2, "11299": 2, "11222": 2, "11297": 2, "11219": 2, "11292": 2, "11214": 2, "11291": 2, "11213": 2, "11286": 2, "11208": 2, "11289": 2, "11207": 2, "569": 2, "564": 2, "matplotlib": [2, 3, 4], "pyplot": [2, 3, 4], "plt": [2, 3, 4], "plot": [2, 3, 4, 5], "info": [2, 3, 4], "scipi": [2, 3, 4], "savemat": [2, 3, 4], "do_compress": [2, 3, 4], "savez_compress": [2, 3, 4], "motor_pizza": 2, "npz": [2, 3, 4], "els": [2, 3], "lt": [2, 6], "libngpi": 2, "_mesh": 2, "array_class": 2, "meshpoint_class": 2, "pointindex": 2, "0x183dabeeaf0": 2, "gt": [2, 3, 6], "fals": 3, "998": [3, 4], "92": [3, 4], "42": [3, 4], "94": [3, 4], "airl": [3, 4], "airr": [3, 4], "transx": [3, 4], "translat": [3, 4], "vec": [3, 4], "transi": [3, 4], "00028744": [3, 4], "y": [3, 4, 5], "rotstat": [3, 4], "x": [3, 4, 5, 6], "len": [3, 4], "nameerror": 3, "traceback": 3, "most": 3, "recent": 3, "call": 3, "last": 3, "cell": 3, "In": 3, "112": 3, "114": 3, "117": 3, "119": 3, "39": [3, 6], "e": [3, 5], "v0": 3, "v1": 3, "size": [3, 5], "index": 3, "argwher": [3, 5], "delet": 3, "n": [3, 5], "motor_pizza_gap": 3, "motor_gap": 3, "34": [3, 5], "13": 3, "boundary_edg": 3, "pizza2": 4, "sqrt": 4, "135": 4, "geo1": 4, "geo2": 4, "305": 4, "motor_pizza_withslic": 4, "befor": 5, "solv": 5, "non": 5, "linear": 5, "magnetostat": [0, 5], "first": 5, "flow": 5, "insid": 5, "omega_c": 5, "The": 5, "properti": 5, "solenoid": 5, "further": 5, "presum": 5, "exist": 5, "potenti": 5, "phi": 5, "sigma": 5, "nabla": 5, "denot": 5, "connect": 5, "As": 5, "boundari": 5, "condit": 5, "prescrib": 5, "cdot": 5, "partial_n": 5, "exterior": 5, "gamma_": 5, "ex": 5, "inflow": 5, "outflow": 5, "out": 5, "respect": 5, "altogeth": 5, "have": 5, "begin": 5, "align": 5, "qquad": 5, "text": 5, "end": 5, "howev": 5, "our": 5, "case": 5, "loop": 5, "come": 5, "plai": 5, "For": 5, "purpos": 5, "introduc": 5, "fictiti": 5, "clone": 5, "order": [5, 6], "abl": 5, "necessari": 5, "load": 5, "previou": 5, "document": 5, "captur": [5, 6], "run": [5, 6], "team_13_geometri": 5, "ipynb": [5, 6], "skspars": 5, "cholmod": 5, "choleski": 5, "chol": [5, 6], "mesh3": [5, 6], "4132": 5, "23369": 5, "nf": 5, "3059": 5, "ne": 5, "720": 5, "nf_all": 5, "46966": 5, "ne_al": 5, "27728": 5, "piec": 5, "code": 5, "below": 5, "duplic": 5, "describ": 5, "abov": 5, "new": 5, "face_index": 5, "tool": [5, 6], "getindic": 5, "f": 5, "boundaryfaces_region": 5, "new_fac": 5, "copi": 5, "points_to_dupl": 5, "uniqu": 5, "ravel": 5, "new_point": 5, "arang": 5, "actual_point": 5, "t_new": 5, "p_new": 5, "f_new": 5, "enumer": 5, "vstack": 5, "tet": 5, "coordin": 5, "contain": 5, "ith": 5, "tets_containing_point": 5, "_": 5, "check": 5, "mp_tet": 5, "tile": 5, "max": 5, "astyp": 5, "regions_2d_new": 5, "stop": 5, "regions_3d": 5, "regions_1d": 5, "4149": 5, "3076": 5, "47028": 5, "27806": 5, "d": [5, 6], "assemble3": [5, 6], "db": 5, "assembleb3": 5, "n1": 5, "n2": 5, "n3": 5, "assemblen3": 5, "unit_coil": 5, "evaluate3": 5, "coeff": 5, "lambda": 5, "region": 5, "face_in_1": 5, "evaluateb3": 5, "diagon": 5, "face_in_2": 5, "face_in_3": 5, "phi_h1": [5, 6], "h1": [5, 6], "space": [5, 6], "p1": [5, 6], "matrix": [5, 6], "dphix_h1": [5, 6], "dphiy_h1": [5, 6], "dphiz_h1": [5, 6], "phib_h1": 5, "r0": [5, 6], "rs0": 5, "assembler3": [5, 6], "rs1": 5, "rz": 5, "removezero": 5, "7e6": 5, "solve_a": [5, 6], "dx_x": [5, 6], "dy_x": [5, 6], "dz_x": [5, 6], "dphix_h1_p0": [5, 6], "dphiy_h1_p0": [5, 6], "dphiz_h1_p0": [5, 6], "unit_coil_p0": 5, "dx_x_p0": 5, "dy_x_p0": 5, "dz_x_p0": 5, "grid": [5, 6], "vtklib": [5, 6], "createvtk": [5, 6], "add_h1_scalar": [5, 6], "add_l2_vector": [5, 6], "grad_j": [5, 6], "writevtk": [5, 6], "current_dens": [5, 6], "vtu": [5, 6], "108": [], "pyvista": [5, 6], "pv": [5, 6], "read": [5, 6], "headerdata": 5, "unstructuredgridinform": 5, "cells23369": 5, "points4149": 5, "bound": 5, "000e": 5, "02": 5, "arrays3": 5, "namefieldtypen": 5, "compminmax": 5, "jpointsfloat3210": 5, "001": 5, "00": 5, "scalars_cellsfloat6410": 5, "005": 5, "grad_jcellsfloat323": 5, "123e": 5, "032": 5, "110e": 5, "03": [5, 6], "123": [], "jupyter_backend": [5, 6], "html": [5, 6], "export_html": 5, "kek": 5, "add_mesh": [5, 6], "style": [5, 6], "wirefram": 5, "blue": 5, "label": [5, 6], "none": [5, 6], "clip_scalar": 5, "scalar": 5, "scalars_": [5, 6], "valu": 5, "invert": 5, "sample_funct": 5, "nois": 5, "threshold": [5, 6], "add_legend": 5, "plotter": [5, 6], "opac": [5, 6], "show_ax": 5, "camera_posit": [5, 6], "93": 5, "thresh": [5, 6], "show_edg": 5, "w": [5, 6], "set_active_scalar": [5, 6], "surfac": [5, 6], "set_active_vector": [5, 6], "arrow": [5, 6], "glyph": [5, 6], "scale": [5, 6], "orient": [5, 6], "toler": [5, 6], "factor": [5, 6], "9500": [5, 6], "black": [5, 6], "polydatainform": [], "cells10245": [], "points21173": [], "strips0": [], "929e": [], "938e": [], "930e": [], "937e": [], "695e": [], "01": [], "645e": [], "scalars_pointsfloat6410": [], "glyphscalepointsfloat3210": [], "002": [], "149e": [], "glyphvectorpointsfloat323": [], "018e": [], "031": [], "965e": [], "790": [], "10000": [], "comput": 0, "current": 0, "densiti": 0, "j": 0, "curl": 5, "operatornam": 5, "div": 5, "18": [], "tree": 6, "cotre": 6, "gaug": 6, "tree_cotree_gaug": 6, "19": [], "27728x23597": 6, "spars": 6, "class": 6, "float64": 6, "23597": 6, "store": 6, "element": 6, "compress": 6, "column": 6, "format": 6, "assembl": 6, "rss": 6, "kn": 6, "phix_hcurl": 6, "phiy_hcurl": 6, "phiz_hcurl": 6, "hcurl": 6, "n0": 6, "curlphix_hcurl": 6, "curlphiy_hcurl": 6, "curlphiz_hcurl": 6, "m_hcurl": 6, "k_hcurl": 6, "c_hcurl_h1": 6, "curlphix_hcurl_p0": 6, "curlphiy_hcurl_p0": 6, "curlphiz_hcurl_p0": 6, "phix_hcurl_p0": 6, "phiy_hcurl_p0": 6, "phiz_hcurl_p0": 6, "kr": 6, "mr": 6, "cholkr": 6, "ux": 6, "uy": 6, "uz": 6, "c": 6, "user": 6, "radu": 6, "appdata": 6, "local": 6, "temp": 6, "ipykernel_16768": [], "1558448441": 6, "py": 6, "47": 6, "cholmodtypeconversionwarn": 6, "convert": 6, "csr_matrix": 6, "csc": 6, "vtk": 6, "grad_x": 6, "magnetostatics_solut": 6, "points4132": [], "xpointsfloat321": [], "487e": [], "022": [], "183e": [], "grad_xcellsfloat323": [], "862e": [], "023": [], "299e": [], "1000": 6, "solut": 0, "ipykernel_848": 6, "mesh2": 6, "orang": 6, "arrows2": 6, "400": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"welcom": 0, "fem": 0, "s": 0, "document": 0, "team": [0, 1], "13": [0, 1], "creat": 1, "3d": 1, "geometri": [1, 2, 3, 4], "problem": 1, "mesh": [2, 3, 4], "motor": [2, 3, 4], "comput": 5, "current": 5, "densiti": 5, "j": 5, "coil": 5, "def": [], "curl": [], "operatornam": [], "div": [], "magnetostat": 6, "solut": 6}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.todo": 2, "nbsphinx": 4, "sphinx": 56}})