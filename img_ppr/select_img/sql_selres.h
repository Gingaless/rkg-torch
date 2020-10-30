#ifndef RKG_SQL_SELRES_H
#define RKG_SQL_SELRES_H

#include <fstream>
//#include <jsoncpp/json/json.h>
#include "sqlite3.h"
#include <boost/format.hpp>
#include <stdarg.h>
#include <iostream>


extern const char* RKG_CREATE_TABLE;

extern const char* RKG_INSERT_LOG;

extern const char* RKG_CHECK_DUPPLICATED;

extern const char* RKG_CHECK_DUPPLICATED_DEST_IS_NULL;


class RKG_SELRES_MANAGER
{
    public:
        RKG_SELRES_MANAGER(const char* db_path, const char* tbl_name, bool reconstruct_table = false);
        RKG_SELRES_MANAGER();
        virtual ~RKG_SELRES_MANAGER();
        void insert_log(const char* src, const char* dest, bool selected, const char* response);
        void insert_log(std::string src, std::string dest, bool selected, const char* response);
        bool check_dupplicated(const char* src, const char* dest);
        bool check_dupplicated(std::string src, std::string dest);

    protected:
        void create_table(const char* tbl_name);

    sqlite3* db;
    sqlite3_stmt* pStmtInsertLog;
    sqlite3_stmt* pStmtCheckDupplicated;
    sqlite3_stmt* pStmtCheckDupplicated_dest_is_null;
};

class RKG_SQL_EXCEPTION: public std::exception
{
    public: 
        explicit RKG_SQL_EXCEPTION(sqlite3* db);
        RKG_SQL_EXCEPTION(sqlite3* db, const char* msg);
        RKG_SQL_EXCEPTION(sqlite3* db, boost::format msg_format, const char* args_format, ...);
        virtual ~RKG_SQL_EXCEPTION() noexcept;
        virtual const char* what() const noexcept {return msg_.c_str();}

    protected:
    
    std::string msg_;
};

#endif //RKG_SQL_SELRES_H