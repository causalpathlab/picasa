"""Color palettes adapted from scanpy 
https://github.com/scverse/scanpy/blob/master/scanpy/plotting/palettes.py

and distinct colors python

https://github.com/alan-turing-institute/distinctipy

."""
from matplotlib import cm, colors

colors_10 = list(map(colors.to_hex, cm.tab10.colors))
colors_10[2] = "#279e68"  # green
colors_10[4] = "#aa40fc"  # purple
colors_10[8] = "#b5bd61"  # kakhi

# see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
colors_20 = list(map(colors.to_hex, cm.tab20.colors))

colors_20 = [
    *colors_20[0:14:2],
    *colors_20[16::2],
    *colors_20[1:15:2],
    *colors_20[17::2],
    "#ad494a",
    "#8c6d31",
]

# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# update 1
# orig reference https://research.wu.ac.at/en/publications/escaping-rgbland-selecting-colors-for-statistical-graphics-26
colors_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    "#7f7f7f",
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

colors_24 = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ff7f00",  # Orange
    "#a65628",  # Brown
    "#f781bf",  # Pink
    "#999999",  # Grey
    "#66c2a5",  # Teal
    "#fc8d62",  # Salmon
    "#8da0cb",  # Light Blue
    "#e78ac3",  # Light Purple
    "#a6d854",  # Lime Green
    "#ffd92f",  # Light Yellow
    "#e5c494",  # Light Brown
    "#b3b3b3",  # Silver
    "#8dd3c7",  # Aqua
    "#bebada",  # Lavender
    "#fb8072",  # Coral
    "#80b1d3",  # Sky Blue
    "#fdb462",  # Peach
    "#b3de69",  # Mint
    "#fccde5",  # Pale Pink
    "#d9d9d9"   # Light Grey
]

colors_36 = [
    '#f28d1a', '#764dc8', '#dc143c', '#1f7a8c', '#57a34b', '#482173', '#c4a107', '#ff6600',
    '#ffa07a', '#708090', '#9370db', '#5f9ea0', '#4682b4', '#dda0dd', '#778899', '#98fb98',
    '#b8860b', '#cd5c5c', '#8b4513', '#fa8072', '#a52a2a', '#ff4500', '#daa520', '#8fbc8f',
    '#6b8e23', '#4682b4', '#ff1493', '#7fffd4', '#00ff7f', '#006400', '#0000ff', '#9acd32',
    '#1e90ff', '#800080', '#adff2f', '#ff69b4'
]

# from https://godsnotwheregodsnot.blogspot.com/2012/09/color-distribution-methodology.html
colors_100 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]

def get_colors(N):
    if N<=10:
        return colors_10
    elif N<=20:
        return colors_20
    elif N<=28:
        return colors_28
    elif N<=100:
        return colors_100
        