


process data_rooting {
    input:
        val NAME from params.data_type
        file PATH from data
    output:
        set val("$NAME"), file("train"), file("validation")
        set val("$NAME"), file("validation"), file("testing")
    script:
        if( name == "cam" ) {
            """
            python $CWD/src/python/datasets/spliting_camelyon.py --path $PATH
            """
            
        }
        else {
            println "Not implemented"
        }
}

