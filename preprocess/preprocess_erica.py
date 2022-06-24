import numpy as np
import os
import json
from src.transformers import BertTokenizer, RobertaTokenizer
import random

MODEL_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
}

# Large ERICA rel2id dict:
rel2id = {"P1464": 0, "P150": 1, "P1792": 2, "P1151": 3, "P194": 4, "P1791": 5, "P2959": 6, "P131": 7, "P2184": 8,
          "P237": 9, "P571": 10, "P38": 11, "P17": 12, "P30": 13, "P31": 14, "P36": 15, "P37": 16, "P361": 17,
          "P910": 18, "P1740": 19, "P421": 20, "P1343": 21, "P208": 22, "P1465": 23, "P47": 24, "P1313": 25, "P530": 26,
          "P138": 27, "P78": 28, "P2633": 29, "P610": 30, "P209": 31, "P122": 32, "P85": 33, "P2936": 34, "P2852": 35,
          "P6": 36, "P463": 37, "P35": 38, "P163": 39, "P1906": 40, "P1589": 41, "P155": 42, "P706": 43, "P2853": 44,
          "P1622": 45, "P206": 46, "P1546": 47, "P172": 48, "P972": 49, "P140": 50, "P460": 51, "P279": 52, "P1552": 53,
          "P2579": 54, "P527": 55, "P461": 56, "P607": 57, "P1412": 58, "P119": 59, "P19": 60, "P3373": 61, "P509": 62,
          "P26": 63, "P106": 64, "P103": 65, "P102": 66, "P25": 67, "P27": 68, "P21": 69, "P20": 70, "P22": 71,
          "P1038": 72, "P800": 73, "P570": 74, "P39": 75, "P569": 76, "P241": 77, "P166": 78, "P410": 79, "P551": 80,
          "P734": 81, "P735": 82, "P1196": 83, "P136": 84, "P1411": 85, "P1303": 86, "P108": 87, "P40": 88, "P69": 89,
          "P832": 90, "P1889": 91, "P92": 92, "P417": 93, "P1365": 94, "P457": 95, "P190": 96, "P112": 97, "P793": 98,
          "P1376": 99, "P501": 100, "P1249": 101, "P485": 102, "P1456": 103, "P2670": 104, "P277": 105, "P275": 106,
          "P2701": 107, "P178": 108, "P306": 109, "P2992": 110, "P50": 111, "P462": 112, "P1582": 113, "P418": 114,
          "P2319": 115, "P1424": 116, "P156": 117, "P641": 118, "P1403": 119, "P171": 120, "P141": 121, "P105": 122,
          "P1018": 123, "P737": 124, "P740": 125, "P135": 126, "P170": 127, "P495": 128, "P282": 129, "P2596": 130,
          "P3969": 131, "P366": 132, "P4132": 133, "P2989": 134, "P1304": 135, "P1542": 136, "P1336": 137, "P144": 138,
          "P807": 139, "P1999": 140, "P3161": 141, "P3103": 142, "P61": 143, "P397": 144, "P575": 145, "P186": 146,
          "P1399": 147, "P1853": 148, "P552": 149, "P53": 150, "P937": 151, "P355": 152, "P414": 153, "P159": 154,
          "P452": 155, "P1454": 156, "P127": 157, "P169": 158, "P1056": 159, "P2031": 160, "P802": 161, "P1066": 162,
          "P2032": 163, "P1455": 164, "P1636": 165, "P1050": 166, "P1962": 167, "P137": 168, "P408": 169, "P101": 170,
          "P398": 171, "P411": 172, "P3095": 173, "P97": 174, "P1142": 175, "P828": 176, "P276": 177, "P1478": 178,
          "P580": 179, "P582": 180, "P710": 181, "P598": 182, "P1950": 183, "P1344": 184, "P2578": 185, "P2737": 186,
          "P358": 187, "P1037": 188, "P264": 189, "P2439": 190, "P180": 191, "P708": 192, "P1366": 193, "P407": 194,
          "P577": 195, "P98": 196, "P872": 197, "P674": 198, "P364": 199, "P576": 200, "P65": 201, "P59": 202,
          "P451": 203, "P1576": 204, "P974": 205, "P205": 206, "P1444": 207, "P3096": 208, "P1427": 209, "P4290": 210,
          "P196": 211, "P1389": 212, "P403": 213, "P469": 214, "P1200": 215, "P921": 216, "P1434": 217, "P2360": 218,
          "P885": 219, "P1299": 220, "P1431": 221, "P2061": 222, "P915": 223, "P449": 224, "P161": 225, "P1811": 226,
          "P750": 227, "P189": 228, "P1672": 229, "P1383": 230, "P2238": 231, "P3716": 232, "P2348": 233, "P749": 234,
          "P1072": 235, "P2175": 236, "P4552": 237, "P210": 238, "P58": 239, "P57": 240, "P344": 241, "P86": 242,
          "P511": 243, "P2632": 244, "P157": 245, "P488": 246, "P2872": 247, "P2289": 248, "P2286": 249, "P927": 250,
          "P2789": 251, "P197": 252, "P3610": 253, "P669": 254, "P81": 255, "P84": 256, "P1619": 257, "P1192": 258,
          "P556": 259, "P2849": 260, "P1049": 261, "P412": 262, "P512": 263, "P263": 264, "P301": 265, "P1435": 266,
          "P2614": 267, "P184": 268, "P3018": 269, "P200": 270, "P201": 271, "P2359": 272, "P2358": 273, "P2366": 274,
          "P2365": 275, "P1879": 276, "P2564": 277, "P149": 278, "P825": 279, "P2416": 280, "P1441": 281, "P1080": 282,
          "P2354": 283, "P413": 284, "P54": 285, "P115": 286, "P286": 287, "P118": 288, "P585": 289, "P2848": 290,
          "P1269": 291, "P840": 292, "P272": 293, "P942": 294, "P655": 295, "P110": 296, "P123": 297, "P941": 298,
          "P450": 299, "P1830": 300, "P859": 301, "P3966": 302, "P437": 303, "P179": 304, "P852": 305, "P400": 306,
          "P404": 307, "P287": 308, "P943": 309, "P908": 310, "P479": 311, "P914": 312, "P375": 313, "P619": 314,
          "P1145": 315, "P176": 316, "P522": 317, "P618": 318, "P1876": 319, "P621": 320, "P1416": 321, "P175": 322,
          "P676": 323, "P720": 324, "P1283": 325, "P2283": 326, "P797": 327, "P3719": 328, "P636": 329, "P2079": 330,
          "P559": 331, "P162": 332, "P1981": 333, "P971": 334, "P4224": 335, "P2152": 336, "P517": 337, "P2375": 338,
          "P814": 339, "P177": 340, "P2868": 341, "P193": 342, "P4149": 343, "P515": 344, "P729": 345, "P611": 346,
          "P2408": 347, "P725": 348, "P360": 349, "P703": 350, "P1995": 351, "P3022": 352, "P944": 353, "P1040": 354,
          "P562": 355, "P3450": 356, "P1547": 357, "P91": 358, "P664": 359, "P532": 360, "P3300": 361, "P1064": 362,
          "P1875": 363, "P1290": 364, "P121": 365, "P1532": 366, "P1881": 367, "P2541": 368, "P1479": 369, "P606": 370,
          "P516": 371, "P1654": 372, "P2341": 373, "P185": 374, "P1001": 375, "P747": 376, "P629": 377, "P126": 378,
          "P730": 379, "P111": 380, "P3876": 381, "P1075": 382, "P1535": 383, "P543": 384, "P542": 385, "P2554": 386,
          "P545": 387, "P541": 388, "P991": 389, "P466": 390, "P913": 391, "P113": 392, "P114": 393, "P2388": 394,
          "P931": 395, "P609": 396, "P16": 397, "P134": 398, "P88": 399, "P3512": 400, "P2643": 401, "P2500": 402,
          "P2846": 403, "P289": 404, "P3094": 405, "P741": 406, "P371": 407, "P1346": 408, "P3828": 409, "P3179": 410,
          "P880": 411, "P3113": 412, "P291": 413, "P2176": 414, "P2293": 415, "P945": 416, "P770": 417, "P167": 418,
          "P1073": 419, "P1433": 420, "P1880": 421, "P4312": 422, "P3602": 423, "P3342": 424, "P3764": 425, "P826": 426,
          "P870": 427, "P1191": 428, "P4000": 429, "P689": 430, "P2505": 431, "P195": 432, "P608": 433, "P1071": 434,
          "P2746": 435, "P2962": 436, "P881": 437, "P690": 438, "P579": 439, "P4195": 440, "P1557": 441, "P3438": 442,
          "P1308": 443, "P748": 444, "P3650": 445, "P523": 446, "P524": 447, "P822": 448, "P2700": 449, "P66": 450,
          "P739": 451, "P3137": 452, "P3349": 453, "P3975": 454, "P1387": 455, "P1317": 456, "P912": 457, "P1026": 458,
          "P2512": 459, "P1462": 460, "P4292": 461, "P1596": 462, "P3448": 463, "P2978": 464, "P1598": 465, "P841": 466,
          "P1029": 467, "P930": 468, "P2515": 469, "P2747": 470, "P3402": 471, "P1877": 472, "P3403": 473, "P3205": 474,
          "P183": 475, "P3301": 476, "P658": 477, "P2517": 478, "P1027": 479, "P520": 480, "P2817": 481, "P2851": 482,
          "P1640": 483, "P1302": 484, "P1419": 485, "P1884": 486, "P1340": 487, "P553": 488, "P3858": 489, "P2363": 490,
          "P2377": 491, "P427": 492, "P3489": 493, "P129": 494, "P1420": 495, "P1429": 496, "P2094": 497, "P423": 498,
          "P647": 499, "P376": 500, "P566": 501, "P3156": 502, "P406": 503, "P2445": 504, "P425": 505, "P3912": 506,
          "P837": 507, "P1034": 508, "P199": 509, "P1716": 510, "P1923": 511, "P3085": 512, "P1406": 513, "P785": 514,
          "P783": 515, "P787": 516, "P786": 517, "P784": 518, "P789": 519, "P788": 520, "P2650": 521, "P3306": 522,
          "P483": 523, "P3780": 524, "P2813": 525, "P504": 526, "P4220": 527, "P1754": 528, "P4185": 529, "P1678": 530,
          "P1322": 531, "P1312": 532, "P3174": 533, "P2738": 534, "P853": 535, "P87": 536, "P3823": 537, "P3080": 538,
          "P1414": 539, "P467": 540, "P833": 541, "P2894": 542, "P1033": 543, "P769": 544, "P1657": 545, "P2922": 546,
          "P3893": 547, "P2925": 548, "P1885": 549, "P2743": 550, "P631": 551, "P3919": 552, "P3279": 553, "P620": 554,
          "P1074": 555, "P3320": 556, "P4100": 557, "P1046": 558, "P2522": 559, "P1533": 560, "P1560": 561,
          "P1165": 562, "P2546": 563, "P1423": 564, "P533": 565, "P2629": 566, "P2756": 567, "P624": 568, "P2389": 569,
          "P1318": 570, "P1731": 571, "P1321": 572, "P3679": 573, "P2563": 574, "P812": 575, "P2033": 576, "P1347": 577,
          "P500": 578, "P1382": 579, "P1327": 580, "P2597": 581, "P3075": 582, "P3033": 583, "P2318": 584, "P3091": 585,
          "P2545": 586, "P780": 587, "P2821": 588, "P4330": 589, "P489": 590, "P2647": 591, "P534": 592, "P3815": 593,
          "P3189": 594, "P3310": 595, "P2329": 596, "P547": 597, "P2291": 598, "P2974": 599, "P1840": 600, "P1891": 601,
          "P3803": 602, "P693": 603, "P634": 604, "P2695": 605, "P744": 606, "P767": 607, "P2684": 608, "P2239": 609,
          "P1002": 610, "P694": 611, "P726": 612, "P1951": 613, "P1750": 614, "P2882": 615, "P2095": 616, "P765": 617,
          "P1136": 618, "P2975": 619, "P3263": 620, "P2669": 621, "P2827": 622, "P682": 623, "P680": 624, "P681": 625,
          "P702": 626, "P4379": 627, "P470": 628, "P1568": 629, "P2396": 630, "P1571": 631, "P1851": 632, "P2058": 633,
          "P3025": 634, "P736": 635, "P3195": 636, "P3739": 637, "P3741": 638, "P3701": 639, "P1531": 640, "P2438": 641,
          "P1809": 642, "P923": 643, "P2285": 644, "P2828": 645, "P505": 646, "P1639": 647, "P1537": 648, "P4147": 649,
          "P2417": 650, "P2321": 651, "P2371": 652, "P3415": 653, "P2575": 654, "P3092": 655, "P3842": 656,
          "P2869": 657, "P1268": 658, "P1000": 659, "P1079": 660, "P3822": 661, "P2758": 662, "P751": 663, "P3190": 664,
          "P3416": 665, "P3093": 666, "P1158": 667, "P3818": 668, "P2913": 669, "P248": 670, "P3428": 671, "P805": 672,
          "P924": 673, "P660": 674, "P2499": 675, "P3264": 676, "P1625": 677, "P3259": 678, "P654": 679, "P2384": 680,
          "P3643": 681, "P4202": 682, "P4002": 683, "P2770": 684, "P1028": 685, "P1990": 686, "P3938": 687,
          "P3019": 688, "P1060": 689, "P2155": 690, "P589": 691, "P1878": 692, "P795": 693, "P1032": 694, "P612": 695,
          "P2012": 696, "P518": 697, "P2784": 698, "P2839": 699, "P468": 700, "P834": 701, "P1372": 702, "P3158": 703,
          "P1637": 704, "P1201": 705, "P1068": 706, "P1041": 707, "P622": 708, "P3262": 709, "P1202": 710, "P803": 711,
          "P3275": 712, "P1398": 713, "P1887": 714, "P2860": 715, "P1706": 716, "P1534": 717, "P1035": 718,
          "P2825": 719, "P2462": 720, "P1408": 721, "P688": 722, "P1057": 723, "P684": 724, "P2548": 725, "P746": 726,
          "P1574": 727, "P3833": 728, "P3461": 729, "P128": 730, "P2156": 731, "P642": 732, "P4099": 733, "P1611": 734,
          "P2673": 735, "P2674": 736, "P916": 737, "P3082": 738, "P1656": 739, "P3026": 740, "P399": 741, "P3261": 742,
          "P1445": 743, "P928": 744, "P926": 745, "P970": 746, "P790": 747, "P1319": 748, "P1039": 749, "P546": 750,
          "P1604": 751, "P3150": 752, "P3015": 753, "P2702": 754, "P1956": 755, "P588": 756, "P3816": 757, "P2127": 758,
          "P2288": 759, "P878": 760, "P2157": 761, "P3999": 762, "P2652": 763, "P1620": 764, "P3712": 765, "P1211": 766,
          "P1536": 767, "P831": 768, "P567": 769, "P1652": 770, "P887": 771, "P925": 772, "P3834": 773, "P2838": 774,
          "P2098": 775, "P1924": 776, "P4545": 777, "P816": 778, "P1199": 779, "P1903": 780, "P2376": 781, "P2378": 782,
          "P3005": 783, "P1432": 784, "P823": 785, "P2754": 786, "P1013": 787, "P3014": 788, "P2715": 789, "P1591": 790,
          "P574": 791, "P3491": 792, "P3490": 793, "P568": 794, "P514": 795, "P811": 796, "P3437": 797, "P521": 798,
          "P2937": 799, "P415": 800, "P4428": 801, "P3931": 802, "P2550": 803, "P1704": 804, "P3989": 805, "P3374": 806,
          "P3032": 807, "P1170": 808, "P2392": 809, "P1137": 810, "P3081": 811, "P2637": 812, "P1594": 813,
          "P1592": 814, "P1593": 815, "P1595": 816, "P3484": 817, "P1264": 818, "P3729": 819, "P1817": 820,
          "P1753": 821, "P3173": 822, "P453": 823, "P2881": 824, "P1204": 825, "P2551": 826, "P1078": 827, "P707": 828,
          "P1558": 829, "P1629": 830, "P3734": 831, "P2875": 832, "P2302": 833, "P1855": 834, "P2668": 835, "P813": 836,
          "P1363": 837, "P2453": 838, "P3592": 839, "P1227": 840, "P3447": 841, "P3432": 842, "P3776": 843,
          "P2159": 844, "P3216": 845, "P1605": 846, "P459": 847, "P1775": 848, "P550": 849, "P768": 850, "P2429": 851,
          "P3713": 852, "P1480": 853, "P2210": 854, "P1909": 855, "P3985": 856, "P4345": 857, "P2568": 858,
          "P2964": 859, "P2679": 860, "P2667": 861, "P3709": 862, "P3440": 863, "P2591": 864, "P3501": 865,
          "P2279": 866, "P2739": 867, "P578": 868, "P1326": 869, "P1915": 870, "P967": 871, "P1393": 872, "P3871": 873,
          "P3831": 874, "P2237": 875, "P1012": 876, "P2567": 877, "P4437": 878, "P1660": 879, "P1910": 880,
          "P2682": 881, "P2681": 882, "P3497": 883, "P3494": 884, "P4329": 885, "P2976": 886, "P369": 887, "P2935": 888,
          "P2876": 889, "P659": 890, "P2379": 891, "P756": 892, "P2414": 893, "P4101": 894, "P2634": 895, "P2960": 896,
          "P3354": 897, "P3355": 898, "P3433": 899, "P2308": 900, "P531": 901, "P1773": 902, "P3364": 903, "P3496": 904,
          "P2675": 905, "P447": 906, "P2560": 907, "P4044": 908, "P3781": 909, "P3774": 910, "P1789": 911, "P3148": 912,
          "P4032": 913, "P4424": 914, "P4320": 915, "P4322": 916, "P4323": 917, "P1777": 918, "P4510": 919,
          "P1011": 920, "P868": 921, "P538": 922, "P565": 923, "P873": 924, "P2587": 925, "P3902": 926, "P4425": 927,
          "P2822": 928, "P4426": 929, "P3493": 930, "P4082": 931, "P4151": 932, "P1349": 933, "P1310": 934,
          "P1221": 935, "P817": 936, "P4043": 937, "P1686": 938, "P2841": 939, "P2353": 940, "P560": 941, "P2305": 942,
          "P2553": 943, "P4006": 944, "P1779": 945, "P2502": 946, "P2303": 947, "P2501": 948, "P1425": 949,
          "P2680": 950, "P4324": 951, "P3578": 952, "P1776": 953, "P3680": 954, "P922": 955, "P4353": 956, "P1210": 957,
          "P2831": 958, "P3037": 959, "P1703": 960, "P3460": 961, "P1917": 962, "P1912": 963, "P2352": 964,
          "P2820": 965, "P3772": 966, "P4443": 967, "P4444": 968, "P4446": 969, "P4387": 970, "P3967": 971, "P537": 972,
          "P1916": 973, "P2577": 974, "P3335": 975, "P1898": 976, "P1897": 977, "P1918": 978, "P1911": 979,
          "P1914": 980, "P1354": 981, "P4321": 982, "P3773": 983, "P3274": 984, "P2507": 985, "P4070": 986,
          "P1606": 987, "P3730": 988, "P1734": 989, "P2361": 990, "P1677": 991, "P3464": 992, "P3356": 993,
          "P2443": 994, "P1171": 995, "P1194": 996, "P794": 997, "P1913": 998, "P4733": 999, "P4614": 1000,
          "P4843": 1001, "P4794": 1002, "P4675": 1003, "P4647": 1004, "P4661": 1005, "P4600": 1006, "P4792": 1007,
          "P4622": 1008, "P4791": 1009, "P4628": 1010, "P4805": 1011, "P4777": 1012, "P4810": 1013, "P4586": 1014,
          "P4688": 1015, "P4584": 1016, "P4743": 1017, "P4774": 1018, "P4873": 1019, "P4875": 1020, "P4844": 1021,
          "P548": 1022, "P4770": 1023, "P4745": 1024, "P1780": 1025, "P4543": 1026, "P4646": 1027, "P4608": 1028,
          "P4634": 1029, "P4599": 1030, "P4882": 1031, "P4809": 1032, "P4624": 1033, "P4850": 1034, "P143": 1035,
          "P3294": 1036, "P2309": 1037, "P4884": 1038, "P1642": 1039}


def preprocess_erica(config):
    raise Exception('Need to change _bert_.npy file names to same either bert or roberta tokens')
    output_dir = config.data_dir + '_preprocessed'
    max_seq_length = config.max_seq_length
    json.dump(rel2id, open(os.path.join(output_dir, 'rel2id.json'), "w"))  # TODO: NOTE -- only need this once

    assert (config.model_type in ['bert', 'roberta'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_files = 10
    file_types = []
    for i in range(n_files):
        file_types.append(f'train_distant_{i}')

    print('--> Preprocessing ERICA pretraining files:', file_types)
    fact_in_annotated_train = set([])

    tokenizer = MODEL_CLASSES[config.model_type].from_pretrained(config.model_name_or_path,
                                                                 do_lower_case=config.do_lower_case)

    def save_data_format(ori_data, is_training, start_uid):
        data = []
        uid = start_uid
        for i in range(len(ori_data)):
            Ls = [0]
            L = 0
            for x in ori_data[i]['sents']:
                L += len(x)
                Ls.append(L)

            vertexSet = ori_data[i]['vertexSet']
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)

            ori_data[i]['vertexSet'] = vertexSet

            item = {}
            item['vertexSet'] = vertexSet
            labels = ori_data[i].get('labels', [])

            train_triple = set([])
            new_labels = []
            for label in labels:
                rel = label['r']
                try:
                    assert rel in rel2id
                except:
                    raise ValueError('rel not in rel2id dict!')
                label['r'] = rel2id[label['r']]
                label['uid'] = uid
                uid += 1

                train_triple.add((label['h'], label['t']))

                label['in_annotated_train'] = False

                if is_training:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            fact_in_annotated_train.add((n1['name'], n2['name'], rel))
                else:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            if (n1['name'], n2['name'], rel) in fact_in_annotated_train:
                                label['in_annotated_train'] = True

                new_labels.append(label)

            item['labels'] = new_labels
            # item['title'] = ori_data[i]['title'] # NOT USED IN TRAINING DATA

            na_triple = []
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet)):
                    if (j != k):
                        if (j, k) not in train_triple:
                            na_triple.append((j, k))

            item['na_triple'] = na_triple
            item['Ls'] = Ls
            item['sents'] = ori_data[i]['sents']
            data.append(item)
        end_uid = uid
        return data, end_uid

    def init(data_file_name, rel2id, config, max_seq_length=max_seq_length, is_training=True, suffix='', start_uid=0):
        ori_data = json.load(open(data_file_name))

        if config.ratio < 1 and suffix == 'train':
            random.shuffle(ori_data)
            print(len(ori_data))
            ori_data = ori_data[: int(config.ratio * len(ori_data))]
            print(len(ori_data))

        data, end_uid = save_data_format(ori_data, is_training, start_uid)
        print('data_len:', len(data))

        print("Saving files")
        if config.ratio < 1 and suffix == 'train':
            json.dump(data, open(os.path.join(output_dir, suffix + '_' + str(config.ratio) + '.json'), "w"))
        else:
            json.dump(data, open(os.path.join(output_dir, suffix + '.json'), "w"))

        sen_tot = len(ori_data)
        bert_token = np.zeros((sen_tot, max_seq_length), dtype=np.int64)
        bert_mask = np.zeros((sen_tot, max_seq_length), dtype=np.int64)
        bert_starts_ends = np.ones((sen_tot, max_seq_length, 2), dtype=np.int64) * (max_seq_length - 1)

        if config.model_type == 'bert':

            for i in range(len(ori_data)):
                item = ori_data[i]
                tokens = []
                for sent in item['sents']:
                    tokens += sent

                subwords = list(map(tokenizer.tokenize, tokens))
                subword_lengths = list(map(len, subwords))
                flatten_subwords = [x for x_list in subwords for x in x_list]

                tokens = [tokenizer.cls_token] + flatten_subwords[: max_seq_length - 2] + [tokenizer.sep_token]
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                token_start_idxs[token_start_idxs >= max_seq_length - 1] = max_seq_length - 1
                token_end_idxs = 1 + np.cumsum(subword_lengths)
                token_end_idxs[token_end_idxs >= max_seq_length - 1] = max_seq_length - 1

                tokens = tokenizer.convert_tokens_to_ids(tokens)
                pad_len = max_seq_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask

                bert_starts_ends[i, :len(subword_lengths), 0] = token_start_idxs
                bert_starts_ends[i, :len(subword_lengths), 1] = token_end_idxs
        else:
            for i in range(len(ori_data)):
                item = ori_data[i]
                words = []
                for sent in item['sents']:
                    words += sent

                idxs = []
                text = ""
                for word in words:
                    if len(text) > 0:
                        text = text + " "
                    idxs.append(len(text))
                    text += word.lower()

                subwords = tokenizer.tokenize(text)

                char2subwords = []
                L = 0
                sub_idx = 0
                L_subwords = len(subwords)
                while sub_idx < L_subwords:
                    subword_list = []
                    prev_sub_idx = sub_idx
                    while sub_idx < L_subwords:
                        subword_list.append(subwords[sub_idx])
                        sub_idx += 1

                        subword = tokenizer.convert_tokens_to_string(subword_list)
                        sub_l = len(subword)
                        if text[L:L + sub_l] == subword:
                            break

                    assert text[L:L + sub_l] == subword

                    char2subwords.extend([prev_sub_idx] * sub_l)

                    L += len(subword)

                if len(text) > len(char2subwords):
                    text = text[:len(char2subwords)]

                assert (len(text) == len(char2subwords))
                tokens = [tokenizer.cls_token] + subwords[: max_seq_length - 2] + [tokenizer.sep_token]

                L_ori = len(tokens)
                tokens = tokenizer.convert_tokens_to_ids(tokens)

                pad_len = max_seq_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask

                for j in range(len(words)):
                    idx = char2subwords[idxs[j]] + 1
                    idx = min(idx, max_seq_length - 1)

                    x = idxs[j] + len(words[j])
                    if x == len(char2subwords):
                        idx2 = L_ori
                    else:
                        idx2 = char2subwords[x] + 1
                        idx2 = min(idx2, max_seq_length - 1)

                    bert_starts_ends[i][j][0] = idx
                    bert_starts_ends[i][j][1] = idx2

        print("Finishing processing")
        if config.ratio < 1 and suffix == 'train':
            np.save(os.path.join(output_dir, suffix + '_' + str(config.ratio) + '_bert_token.npy'), bert_token)
            np.save(os.path.join(output_dir, suffix + '_' + str(config.ratio) + '_bert_mask.npy'), bert_mask)
            np.save(os.path.join(output_dir, suffix + '_' + str(config.ratio) + '_bert_starts_ends.npy'),
                    bert_starts_ends)
        else:
            np.save(os.path.join(output_dir, suffix + '_bert_token.npy'), bert_token)
            np.save(os.path.join(output_dir, suffix + '_bert_mask.npy'), bert_mask)
            np.save(os.path.join(output_dir, suffix + '_bert_starts_ends.npy'), bert_starts_ends)
        print("Finish saving")

        return end_uid

    start_uid = 0
    for file_type in file_types:
        print(f'{file_type} start UID: ', start_uid)
        is_training = ('train' in file_type)
        file_name = os.path.join(config.data_dir, f'{file_type}.json')
        start_uid += init(file_name, rel2id, config, max_seq_length=max_seq_length, is_training=is_training,
                          suffix=file_type, start_uid=start_uid)
