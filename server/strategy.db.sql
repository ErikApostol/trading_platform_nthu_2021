create table user (
    user_id integer primary key autoincrement,
    username text unique not null,
    password text not null
);

create table strategy (
strategy_id integer primary key autoincrement,
author text not null,
strategy_name text,
create_date date not null,
return float,
sharpe_ratio float,
max_drawdown float
);

create table assets_in_strategy (
strategy_id integer not null,
asset_ticker text not null,
weight float
);

create table comment (
comment_id integer primary key autoincrement,
author text not null,
date date,
strategy_id integer not null,
comment text
);
