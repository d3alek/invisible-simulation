Feature: angle of polarization 
    Always perpendicular to line connecting the observed to the sun

    Scenario: angle is horizonal when looking at sun
        Given the sun is at altitude horizon 
        and the sun is at east 
        and we look at altitude horizon
        and we look at east
        then angle is horizontal

    Scenario: angle is horizontal when looking at anti-sun
        Given the sun is at altitude horizon 
        and the sun is at east 
        and we look at altitude horizon
        and we look at west 
        then angle is horizontal

    Scenario: vertical at horizon on sunrize when looking south
        Given the sun is at altitude horizon 
        and the sun is at east 
        and we look at altitude horizon
        and we look at south
        then angle is vertical

    Scenario: vertical at zenith on sunrise when looking south
        Given the sun is at altitude horizon 
        and the sun is at east 
        and we look at altitude zenith 
        and we look at south
        then angle is vertical

    Scenario: horizontal at 45 on sunrise when looking east
        Given the sun is at altitude horizon 
        and the sun is at east 
        and we look at altitude 45
        and we look at east 
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking east
        Given the sun is at altitude zenith 
        and we look at altitude horizon
        and we look at east 
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking south
        Given the sun is at altitude zenith 
        and we look at altitude horizon
        and we look at south
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking west
        Given the sun is at altitude zenith 
        and we look at altitude horizon
        and we look at west
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking north
        Given the sun is at altitude zenith 
        and we look at altitude horizon
        and we look at north
        then angle is horizontal
