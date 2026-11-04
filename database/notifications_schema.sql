create table if not exists notifications (
  id uuid primary key default gen_random_uuid(),
  owner_user_id text not null references "user"(id) on delete cascade,
  event_key text not null unique,
  kind text not null,
  status text not null,
  target_path text not null,
  created_at timestamptz not null default now(),
  read_at timestamptz
);
create index if not exists notifications_owner_created on notifications(owner_user_id,created_at desc);
create index if not exists notifications_owner_unread on notifications(owner_user_id,created_at desc) where read_at is null;
create index if not exists notifications_created on notifications(created_at);
