Class GraphKB.Entity Extends %Persistent
{

Property docid As %Library.String(MAXLEN = 200);

Property entityid As %Library.String(MAXLEN = 100);

Property type As %Library.String(MAXLEN = 1000);

ClassMethod LoadData(filepath As %Library.String, purge As %Library.Integer = 1) As %Status
{
    // purge - Whether to clean the table of previously loaded data; default to 1.
    if purge { // purge data
        set myquery = "TRUNCATE TABLE GraphKB.Entity"
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))
        set rset = tStatement.%Execute()
    }

    set myquery = "LOAD DATA FROM FILE '"_filepath_"' INTO GraphKB.Entity USING {""from"":{""file"":{""header"":true}}}"
    set tStatement = ##class(%SQL.Statement).%New()
    $$$ThrowOnError(tStatement.%Prepare(myquery))
    set rset = tStatement.%Execute()
    if (rset.%SQLCODE < 0) {
        throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
    }
    return rset
}

Storage Default
{
<Data name="EntityDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>docid</Value>
</Value>
<Value name="3">
<Value>entityid</Value>
</Value>
<Value name="4">
<Value>type</Value>
</Value>
</Data>
<DataLocation>^GraphKB.EntityD</DataLocation>
<DefaultData>EntityDefaultData</DefaultData>
<ExtentSize>159</ExtentSize>
<IdLocation>^GraphKB.EntityD</IdLocation>
<IndexLocation>^GraphKB.EntityI</IndexLocation>
<Property name="%%CLASSNAME">
<AverageFieldSize>2</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,16,$lb("-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000"),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,21,21),$lb(758198320,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,758198320))</Histogram>
<OutlierSelectivity>.999999:</OutlierSelectivity>
<Selectivity>0.0001%</Selectivity>
</Property>
<Property name="%%ID">
<AverageFieldSize>3</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(1,11,22,33,44,55,66,77,88,99,109,119,129,139,149,159),$lb(1,0,0,0,0,0,0,0,0,0,1,1,1,1,1),"")</Histogram>
<Selectivity>1</Selectivity>
</Property>
<Property name="docid">
<AverageFieldSize>3.5</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" 0"," 10"," 12"," 13"," 15"," 16"," 17"," 18"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9"),$lb(1,2,2,2,2,2,2,1,1,1,1,1,1,1,1),$lb(540016640,805306368,825229312,805306368,838860800,838860800,855638016,855638016,889192448,889192448,905969664,905969664,922746880,922746880,939524096,825753600,838860800,838860800,855638016,855638016,872415232,872415232,889192448,889192448,905969664,905969664,922746880,922746880,939524096,939524096,956301312,540606464))</Histogram>
<Selectivity>5.2632%</Selectivity>
</Property>
<Property name="entityid">
<AverageFieldSize>21.02</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" 50% NEUTRALIZING ANTIBODY TITER"," BAYESIAN EARLY PHASE TRIAL DESIGNS"," CLINICAL TRIAL NLP"," COVID-19 VACCINE TRIAL PARTICIPANT"," DRUG DEVELOPMENT"," HAWKES PROCESSES"," IYA KHALIL"," KAILEIGH GALLAGHER"," MATS STENSRUD"," NLP"," PETER F. THALL"," ROSS M. KEDL"," SMOOTHED PROBABILISTIC PARAFAC MODEL WITH COVARIATES (SPACO)"," TRIALINSTRUCT"," VARIATIONAL AUTOENCODER"," ZIFENG WANG"),$lb(1,1,2,1,1,1,1,1,1,1,1,1,1,1,1),$lb(540356645,892347680,1111578949,1111578949,1129072974,1279872585,1331054916,1129272905,1146246471,1146246471,1212241739,1212241739,1230586144,1230586144,1262569804,1262569804,1296127059,1296127059,1313624064,1313624064,1346720837,1346720837,1380930387,1380930387,1397575503,1397575503,1414678849,1414678849,1447121481,1447121481,1514751557,542787910))</Histogram>
<Selectivity>0.6849%</Selectivity>
</Property>
<Property name="type">
<AverageFieldSize>7.4</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," PAPER"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"),$lb(8,8,8,8,8,1,1,7,7,7,7,7,7,7,7),$lb(541152596,0,0,0,0,0,0,0,0,0,0,1096111176,1346457669,1346457669,1414484041,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,542396240))</Histogram>
<Selectivity>33.3333%</Selectivity>
</Property>
<SQLMap name="IDKEY">
<BlockCount>-4</BlockCount>
</SQLMap>
<StreamLocation>^GraphKB.EntityS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
