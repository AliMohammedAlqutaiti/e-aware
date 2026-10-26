
            df[
                "oxygen_score"
            ]
        )
        /
        3.0
    ).clip(
        0,
        1
    )

    return df


def hsi_class(
    value
):
    if value >= 0.8:
        return "Very High"

    if value >= 0.6:
        return "High"

    if value >= 0.3:
        return "Moderate"

    return "Low"


def habitat_centroid(
    dataframe
):
    if dataframe.empty:
        return None

    threshold = float(
        dataframe[
            "hsi"
        ].quantile(
            0.75
        )
    )

    best_area = dataframe[
        dataframe[
            "hsi"
        ]
        >=
        threshold
    ].copy()

    if best_area.empty:
        best_area = dataframe.nlargest(
            1,
            "hsi"
        ).copy()

    w = np.clip(
        best_area[
            "hsi"
        ].to_numpy(),
        0.01,
        None
    )

    return (
        float(
            np.average(
                best_area[
                    "lat"
                ],
                weights=w
            )
        ),
        float(
            np.average(
                best_area[
                    "lon"
                ],
                weights=w
            )
        )
    )


# ============================================================
# MOVEMENT
# ============================================================
def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):
    r = 6371.0

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dp = math.radians(
        lat2
        -
        lat1
    )

    dl = math.radians(
        lon2
        -
        lon1
    )

    a = (
        math.sin(
            dp
            /
            2
        )
        ** 2
        +
        math.cos(
            p1
        )
        *
        math.cos(
            p2
        )
        *
        math.sin(
            dl
            /
            2
        )
        ** 2
    )

    return (
        2
        *
        r
        *
        math.asin(
            math.sqrt(
                a
            )
        )
    )


def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):
    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dl = math.radians(
        lon2
        -
        lon1
    )

    x = (
        math.sin(
            dl
        )
        *
        math.cos(
            p2
        )
    )

    y = (
        math.cos(
            p1
        )
        *
        math.sin(
            p2
        )
        -
