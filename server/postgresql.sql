CREATE TABLE assets_in_strategy (
  strategy_id integer NOT NULL,
  asset_ticker varchar(20) NOT NULL,
  weight real
);

CREATE TABLE comment (
  comment_id SERIAL PRIMARY KEY,
  strategy_id integer NOT NULL,
  comment varchar(2000),
  author varchar(30),
  date varchar(16)
);

CREATE TABLE strategy (
  strategy_id SERIAL PRIMARY KEY,
  author varchar(30) NOT NULL,
  create_date varchar(26) NOT NULL,
  return real,
  sharpe_ratio real,
  max_drawdown real,
  strategy_name varchar(50),
  volatility real,
  tw integer,
  competition varchar(30),
  hist_returns bytea
);

CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  username text NOT NULL,
  password text NOT NULL
);

