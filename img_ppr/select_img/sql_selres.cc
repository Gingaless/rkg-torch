#include "sql_selres.h"

const char* RKG_CREATE_TABLE = "create table if not exists %s "
"(log_id integer primary key autoincrement, src_path text not null, "
"dest_path text unique, selected boolean not null, response text);";

const char* RKG_INSERT_LOG = "insert into %s (src_path, dest_path, selected, response) values (?, ?, ?, ?);";

const char* RKG_CHECK_DUPPLICATED = "select * from %s where src_path=? and dest_path=?;";

const char* RKG_CHECK_DUPPLICATED_DEST_IS_NULL = "select * from %s where src_path=? and dest_path is null;";

const char* DROP_TABLE = "drop table if exists %s;";

RKG_SELRES_MANAGER::RKG_SELRES_MANAGER()
{}


RKG_SELRES_MANAGER::RKG_SELRES_MANAGER(const char* db_path, const char* tbl_name, bool reconstruct_table)
{
    /*
    std::ifstream config_json_ifs(config_json_path);
    Json::Reader reader;
    Json::Value config_obj;

    reader.parse(config_json_ifs, config_obj);
    
    const char* db_path = config_obj["dbPATH"].asCString();
    const char* tbl_name = config_obj["tableNAME"].asCString();
    */

    int res = sqlite3_open(db_path, &(this->db));
    if (res!=SQLITE_OK)
        throw RKG_SQL_EXCEPTION(db, boost::format("creating the table %s fails in %s."), "ss",tbl_name, db_path);
    /*
    set_resize(cofig_obj);
    */
   std::string _insert_log = (boost::format(RKG_INSERT_LOG) % tbl_name).str();
   std::string _check_dupplicated = (boost::format(RKG_CHECK_DUPPLICATED) % tbl_name).str();
   std::string _check_dupplicated_dest_is_null = (boost::format(RKG_CHECK_DUPPLICATED_DEST_IS_NULL) % tbl_name).str();

   if (reconstruct_table)
   {
       char* err_msg;
       if (sqlite3_exec(this->db, (boost::format(DROP_TABLE) % tbl_name).str().c_str(), 0, 0, &err_msg) != SQLITE_OK)
       {
           std::string err_msg_ = std::string(err_msg);
           throw RKG_SQL_EXCEPTION(db, boost::format("dropping the table %s failed: %s"), "ss", tbl_name, err_msg_.c_str());
           delete err_msg;
       }
       delete err_msg;
   }
   create_table(tbl_name);

   sqlite3_prepare(db, _insert_log.c_str(), -1, &(this->pStmtInsertLog), NULL);
   sqlite3_prepare(db, _check_dupplicated.c_str(), -1, &(this->pStmtCheckDupplicated), NULL);
   sqlite3_prepare(db, _check_dupplicated_dest_is_null.c_str(), -1, &(this->pStmtCheckDupplicated_dest_is_null), NULL);
}

/*
template <const char* db_path, const char* tbl_name, const char* config_json_path, const char* sql_json_path> 
void RKG_SELRES_MANAGER<db_path, tbl_name, config_json_path, sql_json_path>::set_resize(Json::Value json_obj)
{
    Json::Value heights = json_obj["reiszeH"];
    Json::Value widths = json_obj["resizeW"];
    len_Hs = std::size(heights);
    len_Ws = std::size(widths);
    if (len_Hs != len_Ws)
        throw std::invalid_argument("The length of the height array does not equal to that of the width array.");
    this->resize_H = new int[len_Hs];
    this->resize_W = new int[len_Ws];
    for (int i=0; i<len_Hs; i++)
        {resize_H[i] = heights[i]; resize_W[i] = widths[i];}
}
*/

RKG_SELRES_MANAGER::~RKG_SELRES_MANAGER()
{
    sqlite3_finalize(pStmtInsertLog);
    sqlite3_finalize(pStmtCheckDupplicated);
    pStmtCheckDupplicated = NULL;
    pStmtInsertLog = NULL;
    sqlite3_close(db);
}

void RKG_SELRES_MANAGER::create_table(const char* tbl_name)
{
    char * err_msg;
    int rc = sqlite3_exec(db, (boost::format(RKG_CREATE_TABLE) % tbl_name).str().c_str(),0,0,&err_msg);
    if (rc!=SQLITE_OK)
    {
        std::string err_msg_ = std::string(err_msg);
        delete err_msg;
        throw RKG_SQL_EXCEPTION(db, boost::format("creating the table %s fails: %s"), "ss", tbl_name ,err_msg_.c_str());
    }
    delete err_msg;
}

void RKG_SELRES_MANAGER::insert_log(const char* src, const char* dest, bool selected, const char* response)
{
    sqlite3_reset(pStmtInsertLog);
    sqlite3_bind_text(pStmtInsertLog, 1, src, -1, SQLITE_STATIC);
    sqlite3_bind_text(pStmtInsertLog, 2, dest, -1, SQLITE_STATIC);
    sqlite3_bind_int(pStmtInsertLog, 3, (int)selected);
    sqlite3_bind_text(pStmtInsertLog, 4, response, -1, SQLITE_STATIC);
    int rc = sqlite3_step(pStmtInsertLog);
    if (rc != SQLITE_DONE)
     throw RKG_SQL_EXCEPTION(this->db, boost::format("insert execution failed: %s"), "s", sqlite3_errmsg(this->db));
}

void RKG_SELRES_MANAGER::insert_log(std::string src, std::string dest, bool selected, const char* response)
{
    insert_log(src.c_str(), dest.c_str(), selected, response);
}

bool RKG_SELRES_MANAGER::check_dupplicated(const char* src, const char* dest)
{
    if (dest==NULL)
    {
        sqlite3_reset(this->pStmtCheckDupplicated_dest_is_null);
        sqlite3_bind_text(pStmtCheckDupplicated_dest_is_null, 1, src, -1, SQLITE_STATIC);
        if (sqlite3_step(pStmtCheckDupplicated_dest_is_null)==SQLITE_ROW)
            return true;
    }
    sqlite3_reset(this->pStmtCheckDupplicated);
    sqlite3_bind_text(pStmtCheckDupplicated, 1, src, -1, SQLITE_STATIC);
    sqlite3_bind_text(pStmtCheckDupplicated, 2, dest, -1, SQLITE_STATIC);
    if (sqlite3_step(pStmtCheckDupplicated)==SQLITE_ROW)
        return true;
    else
        return false;
}

bool RKG_SELRES_MANAGER::check_dupplicated(std::string src, std::string dest)
{
    return check_dupplicated(src.c_str(), src.c_str());
}



//RKG_SQL_EXCEPTION

RKG_SQL_EXCEPTION::RKG_SQL_EXCEPTION(sqlite3* db, const char* msg) :
msg_(msg)
{
    sqlite3_close(db);
    delete msg;
}

RKG_SQL_EXCEPTION::RKG_SQL_EXCEPTION(sqlite3* db, boost::format msg_format, const char* args_format, ...)
{
    va_list vl;
    //size_t size;
    int i;
    va_start(vl, args_format);

    for (i=0; args_format[i]!='\0'; ++i)
    {
        const char* s = va_arg(vl, const char*);
        msg_format = msg_format % s;
    }

    this->msg_ = msg_format.str();
    sqlite3_close(db);
}

RKG_SQL_EXCEPTION::~RKG_SQL_EXCEPTION()
{}