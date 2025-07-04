Class GraphKB.Query Extends %RegisteredObject
{

ClassMethod Search(userQueryEntity As %String, userQueryEmbedding As %String, graphSearchCount As %Integer = 0, semanticSearchCount As %Integer = 0) As %String
{
    set sc = $$$OK
    if (graphSearchCount <= 0) && (semanticSearchCount <=0) {
        write "Nothing to display. You need to supply at least 1 non-zero count for graph search or semantic search."
        return '$$$OK
    } else {
        write "Displaying at most "_graphSearchCount_" counts of graph search and "_semanticSearchCount_" counts of semantic search."_$CHAR(10)

    }
    try {
        // Set up SQL Query to get whatever fields you want, joined on the embeddings
        if graphSearchCount > 0 { // Graph Search Enabled
            set myEntityQuery = "SELECT TOP ? entity.entityid FROM GraphKB.EntityEmbeddings emb JOIN GraphKB.Entity entity ON emb.ID = entity.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
            set tEntityStatement = ##class(%SQL.Statement).%New()
            $$$ThrowOnError(tEntityStatement.%Prepare(myEntityQuery))

            // Convert query string into embedding
            #; set userQueryEntity = $CLASSMETHOD("GraphKB.EntityEmbeddings","GetEmbeddingsBatchPy",userQuery)
            
            // Execute SQL
            set rsetEntity = tEntityStatement.%Execute(graphSearchCount, userQueryEntity)
            if (rsetEntity.%SQLCODE < 0) {
                throw ##class(%Exception.SQL).CreateFromSQLCODE(rsetEntity.%SQLCODE, rsetEntity.%Message)
            }

            // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
            while rsetEntity.%Next(){
                set entityList = $s($d(entityList):entityList,1:"")_$lb(rsetEntity.%Get("entityid"))    
            }
            
            set myquery = "SELECT TOP ?  docs.docid,'abstract:'||docs.abstract||' authors:'||docs.authors ||' title:'||docs.title ||' published:' ||docs.published as abstract " 
             _ "FROM GraphKB.Relations rel " 
             _ "JOIN GraphKB.Documents docs ON rel.docid = docs.docid " 
             _ "WHERE rel.source %INLIST ? OR rel.target %INLIST ? " 
             _ "GROUP BY rel.docid,rel.source, rel.target "
             _ "ORDER by count(rel.docid) DESC "
             _ "UNION " 
             _ "SELECT TOP ?  docs.docid,'abstract:'||docs.abstract||' authors:'||docs.authors ||' title:'||docs.title ||' published:' ||docs.published as abstract " 
             _ "FROM GraphKB.DocumentsEmbeddings emb " 
             _ "JOIN GraphKB.Documents docs ON emb.ID = docs.ID " 
             _ "ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
            #; SET myquery = 
            #; "SELECT TOP ? docs.docid, " _
            #; "'abstract:' || docs.abstract || ' authors:' || docs.authors || ' title:' || docs.title || ' published:' || docs.published AS abstract " _
            #; "FROM GraphKB.Relations rel " _
            #; "JOIN GraphKB.Documents docs ON rel.docid = docs.docid " _
            #; "WHERE rel.source %INLIST ? OR rel.target %INLIST ? " _
            #; "GROUP BY docs.docid, docs.abstract, docs.authors, docs.title, docs.published " _
            #; "ORDER BY COUNT(*) DESC"



        } elseif semanticSearchCount > 0 { // Semantic search 
            set myquery = "SELECT TOP ? 'abstract:'||docs.abstract||' authors:'||docs.authors ||' title:'||docs.title ||' published:' ||docs.published as abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
            #; set myquery = "SELECT TOP ? 'abstract:'||docs.abstract||' title:'||docs.title as abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"


        }
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))

        // Convert query string into embedding
        #; set userQueryEmbedding = $CLASSMETHOD("GraphKB.DocumentsEmbeddings","GetEmbeddingsBatchPy",userQuery)
        
        // Execute SQL
        //graphSearchCount,userQueryEmbedding
        set rset = $s($d(entityList):tStatement.%Execute(graphSearchCount,entityList,entityList,semanticSearchCount,userQueryEmbedding),1:tStatement.%Execute(semanticSearchCount, userQueryEmbedding))
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
        }
        set retrievedInfo = "", resultCount=0
        // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
        while rset.%Next(){
            set resultCount = resultCount + 1
            set retrievedInfo = retrievedInfo _ $C(13,10) _" Abstract "_resultCount_": " _rset.%Get("abstract")_$CHAR(10)
        }
        write !, retrievedInfo      
        return retrievedInfo

    }
    catch e {
        set sc = e.AsStatus()
        zw sc
        return e.AsSQLMessage()
    }
}

}
