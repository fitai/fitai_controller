-- PostgreSQL database table definitions  --
-- Should only want to run these once,    --
-- but I will leave here for reference of --
-- data types, etc.                       --

-- NOTE:                                  --
-- Table names should be self-explanatory --

CREATE TABLE client_info (
    client_id       BIGINT PRIMARY KEY,
    client_name     VARCHAR(20),
    client_address  TEXT,
    client_signon   date,
    client_misc     TEXT
);
-- INSERT INTO client_info (client_id, client_name, client_signon) VALUES (0, 'NYU_soccer_mens', CURRENT_DATE);

CREATE TABLE athlete_info (
    client_id       BIGINT,
    athlete_id      BIGINT PRIMARY KEY,
    athlete_name    TEXT,
    athlete_age     INT,
    athlete_gender  VARCHAR(5),
    athlete_misc    TEXT
);
-- INSERT INTO athlete_info (client_id, athlete_id, athlete_name, athlete_age, athlete_gender)
--     VALUES (0, 0, 'Kyle Brubaker', 28, 'M');

CREATE TABLE athlete_lift (
    athlete_id          BIGINT,
    lift_id             BIGINT PRIMARY KEY,
    lift_sampling_rate  INT,
    lift_start          TIMESTAMP,
    lift_type           VARCHAR(20),
    lift_weight         INT,
    lift_weight_units   VARCHAR(5),
    lift_num_reps       INT,
    lift_misc           TEXT
);
-- INSERT INTO athlete_lift (athlete_id, lift_id, lift_sampling_rate, lift_start, lift_type, lift_weight, lift_weight_units, lift_num_reps)
--     VALUES (0, 0, 50, CURRENT_TIMESTAMP, 'OHP', 50, 'lbs', 10);


CREATE TABLE lift_data (
    lift_id     BIGINT,
    a_x         DOUBLE PRECISION,
    a_y         DOUBLE PRECISION,
    a_z         DOUBLE PRECISION,
    timepoint   DOUBLE PRECISION,
    CONSTRAINT series_id PRIMARY KEY(lift_id, timepoint)
);

CREATE TABLE lift_data_backup (
    lift_id     BIGINT,
    a_x         DOUBLE PRECISION,
    a_y         DOUBLE PRECISION,
    a_z         DOUBLE PRECISION,
    timepoint   DOUBLE PRECISION,
    CONSTRAINT series_id PRIMARY KEY(lift_id, timepoint)
);

CREATE TABLE lift_data_storage (
    lift_id     BIGINT PRIMARY KEY,
    a_x         FLOAT[],
    a_y         FLOAT[],
    a_z         FLOAT[],
    timepoint   FLOAT[]
);

insert into lift_data_backup(lift_id,a_x,timepoint) select lift_id,a_x,timepoint from lift_data;
