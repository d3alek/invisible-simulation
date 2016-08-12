Feature: angle of polarization 
    Always perpendicular to line connecting the observed to the sun

    Scenario: horizontal at horizon on sunrise looking in every azimuth 1
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at west
        then angle is horizontal

    Scenario: horizontal at horizon on sunrise looking in every azimuth 2
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at north 
        then angle is horizontal

    Scenario: vertical at zenith on sunrise when looking north
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the zenith 
        and we look at north 
        then angle is vertical 

    Scenario: horizontal at zenith on sunrise when looking west 
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the zenith 
        and we look at west
        then angle is horizontal
