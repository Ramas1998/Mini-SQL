import os
import sys
import csv
import numpy as np
import re
import pprint
import itertools
#import pdb


c_oprs = ["<", ">", "<=", ">=", "=", "<>"]
metafile = "../data/metadata.txt"
data_directory = "../data/"
LIT = "<LIT>"
agg_opr = ["max","min","avg", "sum", "distinct","count",]



schema = {}

def error_occrs(x, s):
    if x:
        print("ERROR : {}".format(s))
        exit(-1)

def isint(s):
    try:
        _ = int(s)
        return True
    except:
        return False

def related_oprs(cond):
    if "<=" in cond: op = "<="
    elif ">" in cond: op = ">"
    elif ">=" in cond: op = ">="
    elif "<" in cond: op = "<"
    elif "=" in cond: op = "="
    elif "<>" in cond: op = "<>"
    else : error_occrs(True, "Invalid case : '{}'".format(cond))

    error_occrs(cond.count(op) != 1, "Invalid case : '{}'".format(cond))
    l, r = cond.split(op)
    l = l.strip()
    r = r.strip()
    return op, l, r


def initialise_metadata():
    with open(metafile, "r") as f:
        contents = f.readlines()
    
    contents = [t.strip() for t in contents if t.strip()]

    table_name = None
    for t in contents:
        t = t.lower()
        if t == "<begin_table>": attrs, table_name = [], None
        elif t == "<end_table>": pass
        elif not table_name: table_name, schema[t] = t, []
        else: schema[table_name].append(t)

def table_loading(fname):
    return np.genfromtxt(fname, dtype=int, delimiter=',')

def print_output_table(qdict):
    alias2tb = qdict['alias2tb']
    inter_cols = qdict['inter_cols']
    tables = qdict['tables']
    conditions = qdict['conditions']
    cond_op = qdict['cond_op']
    proj_cols = qdict['proj_cols']

    colidx = {}
    cnt = 0
    all_tables = []
    for t in tables:
        lt = table_loading(os.path.join(data_directory, "{}.csv".format( alias2tb[t] )))

        idxs = [schema[alias2tb[t]].index(cname) for cname in inter_cols[t]]
        lt = lt[:, idxs]
        all_tables.append(lt.tolist())

        colidx[t] = {cname: cnt+i for i, cname in enumerate(inter_cols[t])}
        cnt += len(inter_cols[t])

    inter_table = [[i for tup in r for i in list(tup)] for r in itertools.product(*all_tables)]
    inter_table = np.array(inter_table)

    if len(conditions):
        totake = np.ones((inter_table.shape[0],len(conditions)), dtype=bool)

        for idx, (op, left, right) in enumerate(conditions):
            cols = []
            for tname, cname in [left, right]:
                if tname == LIT: cols.append(np.full((inter_table.shape[0]), int(cname)))
                else: cols.append(inter_table[:, colidx[tname][cname]])

            if op=="<=": totake[:, idx] = (cols[0] <= cols[1])
            if op==">=": totake[:, idx] = (cols[0] >= cols[1])
            if op=="<>": totake[:, idx] = (cols[0] != cols[1])
            if op=="<": totake[:, idx] = (cols[0] < cols[1])
            if op==">": totake[:, idx] = (cols[0] > cols[1])
            if op=="=": totake[:, idx] = (cols[0] == cols[1])

        if cond_op == " or ": final_take = (totake[:, 0] |  totake[:, 1])
        elif cond_op == " and ": final_take = (totake[:, 0] & totake[:, 1])
        else: final_take = totake[:, 0]
        inter_table = inter_table[final_take]

    select_idxs = [colidx[tn][cn] for tn, cn, aggr in proj_cols]
    inter_table = inter_table[:, select_idxs]

    if proj_cols[0][2]:
        out_table = []
        disti = False
        for idx, (tn, cn, aggr) in enumerate(proj_cols):
            col = inter_table[:, idx]
            if aggr == "min": out_table.append(min(col))
            elif aggr == "max": out_table.append(max(col))
            elif aggr == "avg": out_table.append(sum(col)/col.shape[0])
            elif aggr == "count": out_table.append(col.shape[0])
            elif aggr == "sum": out_table.append(sum(col))
            elif aggr == "distinct":
                seen = set()
                out_table = [x for x in col.tolist() if not (x in seen or seen.add(x) )]
                disti = True
            else: error_occrs(True, "Invalid Aggregate_operator")
        out_table = np.array([out_table])
        if disti: out_table = np.array(out_table).T
        out_header = ["{}({}.{})".format(aggr, tn, cn) for tn, cn, aggr in proj_cols]
    else:
        out_table = inter_table
        out_header = ["{}.{}".format(tn, cn) for tn, cn, aggr in proj_cols]
    return out_header, out_table.tolist()

def break_query(q):
    toks = q.lower().split()  # toks : ['select', '*', 'from', 'table1']
    if toks[0] != "select":
        log_error("only select is allowed")

    select_idx = [idx for idx, t in enumerate(toks) if t == "select"] # find the index of select, from and where if present
    from_idx = [idx for idx, t in enumerate(toks) if t == "from"]
    where_idx = [idx for idx, t in enumerate(toks) if t == "where"]

    error_occrs((len(select_idx) != 1) or (len(from_idx) != 1) or (len(where_idx) > 1), "wrong query") # i.e agar length one nhi select_idx,from_idx,where_idx means query wrong hae
    select_idx, from_idx = select_idx[0], from_idx[0] # iniitialisation hua
    where_idx = where_idx[0] if len(where_idx) == 1 else None
    error_occrs(from_idx <= select_idx, "wrong query") # vaise select_idx is 0 and from_idx is 1
    if where_idx: error_occrs(where_idx <= from_idx, "wrong query") # means from where ke bdd ayega else error

    raw_cols = toks[select_idx+1:from_idx] # basically all things from * before from idx
    if where_idx: # agar where query mae hae means conditions
        raw_tables = toks[from_idx+1:where_idx]
        raw_condition = toks[where_idx+1:]
    else:           # no condition then put table name in raw_table with no condition
        raw_tables = toks[from_idx+1:]
        raw_condition = [] 

    error_occrs(len(raw_tables) == 0, "no tables after 'from'") # i.e koi table naam hee nhi likha
    error_occrs(where_idx != None and len(raw_condition) == 0, "no conditions after 'where'") # where likha but koi condition nhi daali
    return raw_tables, raw_cols, raw_condition # return these to parse query

def parsing(raw_tables):
    raw_tables = " ".join(raw_tables).split(",")# split the list of tables
    
    tables = []
    alias2tb = {}
    for rt in raw_tables:
        t = rt.split() # t will contain table1 in first iterate
        error_occrs(not(len(t) == 1 or (len(t) == 3 and t[1] == "as")), "invalid table spacification '{}'".format(rt)) # case like table1 as t1
        if len(t) == 1: tb_name, tb_alias = t[0], t[0]
        else: tb_name, _, tb_alias = t # in case aliasing hui hae toh tb_alias = t1

        error_occrs(tb_name not in schema.keys(), "no table name '{}'".format(tb_name)) # agar table na naam hee nhi hae schema dictionary ame
        error_occrs(tb_alias in alias2tb.keys(), "not unique table/alias '{}'".format(tb_alias))

        tables.append(tb_alias) # now add table name to tables
        alias2tb[tb_alias] = tb_name # and tb_alias ko tb_name sae
    return tables, alias2tb

def parse_proj_columns(raw_cols, tables, alias2tb):

    #pdb.set_trace()
    raw_cols = "".join(raw_cols).split(",") # sare columns ke naam ko list mae daala
    proj_cols = []
    for rc in raw_cols:
        regmatch = re.match("(.+)\((.+)\)", rc) # seeing which column max and min is suppose to be done
        if regmatch: aggr, rc = regmatch.groups() # rc gets the column name
        else: aggr = None

        error_occrs("." in rc and len(rc.split(".")) != 2, "invalid column name '{}'".format(rc))

        tname = None
        if "." in rc:
            tname, cname = rc.split(".")
            error_occrs(tname not in alias2tb.keys(), "unknown field : '{}'".format(rc))
        else:
            cname = rc
            if cname != "*":
                tname = [t for t in tables if cname in schema[alias2tb[t]]]
                error_occrs(len(tname) > 1, "not unique field : '{}'".format(rc))
                error_occrs(len(tname) == 0, "unknown field : '{}'".format(rc))
                tname = tname[0]

        if cname == "*":
            error_occrs(aggr != None, "can't use agg_opr '{}'".format(aggr))
            if tname != None:
                proj_cols.extend([(tname, c, aggr) for c in schema[alias2tb[tname]]])
            else:
                for t in tables:
                    proj_cols.extend([(t, c, aggr) for c in schema[alias2tb[t]]])
        else:
            error_occrs(cname not in schema[alias2tb[tname]], "unknown field : '{}'".format(rc))
            proj_cols.append((tname, cname, aggr))

    s = [a for t, c, a in proj_cols]
    error_occrs(all(s) ^ any(s), "agg_oprd and nonagg_oprd columns are not allowed simultaneously")
    error_occrs(any([(a=="distinct") for a in s]) and len(s)!=1, "distinct can only be used alone")
    
    return proj_cols

def condition_parse(raw_condition, tables, alias2tb):
  
    conditions = []
    cond_op = None
    if raw_condition:
        raw_condition = " ".join(raw_condition)

        if " or " in raw_condition: cond_op = " or "
        elif " and " in raw_condition: cond_op = " and "

        if cond_op: raw_condition = raw_condition.split(cond_op)
        else: raw_condition = [raw_condition]

        for cond in raw_condition:
            relate_op, left, right = related_oprs(cond)
            parsed_cond = [relate_op]
            for idx, rc in enumerate([left, right]):
                if isint(rc):
                    parsed_cond.append((LIT, rc))
                    continue

                if "." in rc:
                    tname, cname = rc.split(".")
                else:
                    cname = rc
                    tname = [t for t in tables if rc in schema[alias2tb[t]]]
                    error_occrs(len(tname) > 1, "not unique field : '{}'".format(rc))
                    error_occrs(len(tname) == 0, "unknown field : '{}'".format(rc))
                    tname = tname[0]
                error_occrs((tname not in alias2tb.keys()) or (cname not in schema[alias2tb[tname]]),
                    "unknown field : '{}'".format(rc))
                parsed_cond.append((tname, cname))
            conditions.append(parsed_cond)
    # ----------------------------------------------
    return conditions, cond_op


def query_parsing(q):

    raw_tables, raw_cols, raw_condition = break_query(q) # raw_tables : name of table,raw_cols mae * and raw_condition mae condition

    tables, alias2tb = parsing(raw_tables) # tables ki list and alias to table ki dictionary

    proj_cols = parse_proj_columns(raw_cols, tables, alias2tb)

    conditions, cond_op = condition_parse(raw_condition, tables, alias2tb)


    inter_cols = {t : set() for t in tables} 
    for tn, cn, _ in proj_cols: inter_cols[tn].add(cn)
    for cond in conditions:
        for tn, cn in cond[1:]:
            if tn == LIT: continue
            inter_cols[tn].add(cn)

    for t in tables: inter_cols[t] = list(inter_cols[t])

    return {
        'tables':tables,
        'alias2tb':alias2tb,
        'proj_cols':proj_cols,
        'conditions':conditions,
        'cond_op':cond_op,
        'inter_cols':inter_cols,
    }


def table_printing(header, table):
    print(",".join(map(str, header)))
    for row in table:
        print(",".join(map(str, row)))


def main():
    initialise_metadata()
    if len(sys.argv) != 2:
        print("ERROR : invalid args")
        print("USAGE : python {} '<sql query>'".format(sys.argv[0]))
        exit(-1)

    
    q = sys.argv[1]

    qdict = query_parsing(q)
    out_header, out_table = print_output_table(qdict)

    table_printing(out_header, out_table)


if __name__ == "__main__":
    main()