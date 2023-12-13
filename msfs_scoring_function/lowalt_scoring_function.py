from numpy import exp, log, floor, maximum, minimum


# To be provided

# SectionTime = 20  # Time spent in section [seconds]
# SectionSumAltitude = 20 * 300  # Integral of altitude over section [feet-seconds]
# SectionAverageAltitude = SectionSumAltitude / SectionTime  # Average altitude over section [feet]
# CurrentSection = 0
# BaseScoreAltitudeMin = 250
# BaseScoreAltitudeMax = 1000
# SectionTimeReference = 10


def get_section_score(
        SectionTime,
        SectionTimeReference,
        SectionAverageAltitude,
        BaseScoreAltitudeMin,
        BaseScoreAltitudeMax,
):

    # Constants:
    BaseScoreMin = 1e4
    BaseScoreMax = 1e5

    # Base Score Calculation (SectionBaseScore):

    dBSA = BaseScoreAltitudeMax - BaseScoreAltitudeMin

    AlphaMax = (  # Note: no relation to angle of attack; just the name MSFS uses
            (log(BaseScoreMin) - log(BaseScoreMax))
            / dBSA
    )
    Alpha = 0.6 * AlphaMax

    Beta = (  # Note: no relation to sideslip angle; just the name MSFS uses
            (exp(dBSA * -Alpha) * BaseScoreMax - BaseScoreMin)
            / (dBSA * BaseScoreMin)
    )

    CorrectedAltitude = maximum(SectionAverageAltitude, BaseScoreAltitudeMin)

    BaseScore = (
            (exp((CorrectedAltitude - BaseScoreAltitudeMin) * -Alpha) * BaseScoreMax)
            / ((CorrectedAltitude - BaseScoreAltitudeMin) * Beta + 1)
    )
    SectionBaseScore = BaseScore

    # Section Score Multiplier Calculation (SectionScoreMultiplier):

    SectionTimeBaseMin = 1.1 * SectionTimeReference
    SectionTimeBaseMax = 4 * SectionTimeReference

    TimeFactor = maximum(
        0,
        minimum(
            1,
            1 - (SectionTime - SectionTimeBaseMin) / (SectionTimeBaseMax - SectionTimeBaseMin)
        )
    )

    SectionScoreMultiplier = 0.75 + 1.25 * TimeFactor

    # Section Score Calculation (SectionScore):
    SectionScore = SectionBaseScore * SectionScoreMultiplier

    return SectionScore
