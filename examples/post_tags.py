from itertools import groupby
from operator import attrgetter

import uvicorn
from databases import Database
from gql import query, gql, field_resolver
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from stargql import GraphQL

from dataloader import DataLoader

schema = gql(
    """
type Query {
    posts(ids: [ID!]): [Post] 
    post(id: ID!): Post
}

type Post {
    id: ID
    name: String
    tags: [Tag]
}

type Tag {
    id: ID
    content: String
}
"""
)

metadata = MetaData()


db = Database('sqlite:///test.db')

Post = Table('post', metadata, Column('id', Integer, primary_key=True), Column('name', String))

Tag = Table(
    'tag',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('content', String),
    Column('post_id', None, ForeignKey('post.id')),
)


def init():
    engine = create_engine('sqlite:///test.db', echo=True)
    metadata.create_all(bind=engine)
    conn = engine.connect()
    conn.execute(
        Post.insert(),
        [{'id': 1, 'name': 'Python'}, {'id': 2, 'name': 'Golang'}, {'id': 3, 'name': 'Java'}],
    )
    conn.execute(
        Tag.insert(),
        [
            {'id': 1, 'content': 'Good', 'post_id': 1},
            {'id': 2, 'content': 'Good', 'post_id': 1},
            {'id': 3, 'content': 'Good', 'post_id': 2},
            {'id': 4, 'content': 'Good', 'post_id': 2},
            {'id': 5, 'content': 'Good', 'post_id': 5},
        ],
    )


async def list_posts(keys):
    s = Post.select().where(Post.c.id.in_(keys))
    return sort(keys, await db.fetch_all(s), key_fn=attrgetter('id'))


def sort(keys, values, key_fn):
    value_map = {key_fn(v): v for v in values}
    return list(map(lambda k: value_map.get(k), keys))


def group_sort(keys, values, *, key):
    value_map = {key: list(data) for key, data in groupby(values, key=key)}
    return list(map(lambda k: value_map.get(k), keys))


async def list_tags_by_post(keys):
    s = Tag.select().where(Tag.c.post_id.in_(keys))
    return group_sort(keys, await db.fetch_all(s), key=attrgetter('post_id'))


loader = DataLoader(list_posts)

tag_loader = DataLoader(list_tags_by_post)


@query
async def posts(parent, info, ids):
    return await db.fetch_all(Post.select().where(Post.c.id.in_(ids)))


@query
async def post(parent, info, id):
    return await loader.load(int(id))


@field_resolver('Post', 'tags')
async def tags(parent, info):
    return await tag_loader.load(parent['id'])


async def startup():
    await db.connect()


async def shutdown():
    await db.disconnect()


app = GraphQL(type_defs=schema, on_shutdown=[shutdown], on_startup=[startup])

if __name__ == '__main__':
    uvicorn.run(app)
