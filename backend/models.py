from __future__ import annotations
from sqlalchemy import Integer, Column, String, Boolean, Enum, UniqueConstraint, ForeignKey
from sqlalchemy.orm import relationship
import enum
from database import Base

class RelationshipType(enum.Enum):
    PENDING = 0
    ACCEPTED = 1
    BLOCKED = 2

class UserRelationship(Base):
    __tablename__ = "UserRelationships"

    user_id = Column(Integer, ForeignKey("users.id"),primary_key=True)
    friend_id = Column(Integer,primary_key=True)
    relationship_status = Column(Enum(RelationshipType))
    UniqueConstraint("user_id", "friend_id", name="unique_friendship")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(20))
    email = Column(String(30), unique=True, index=True)
    hashed_pass = Column(String(30))
    disabled = Column(Boolean)

    relationships = relationship("UserRelationship")

    def add_friend(self, friend: User):

        # if friend has blocked self:

            # return None

        # if friend is friends with self

            # update friend's relationship to have status friends

            # 
        
        # else
        new_relationship = UserRelationship(
            user_id=self.id,
            friend_id=friend.id,
            relationship_status=RelationshipType.PENDING
        )
        self.friends.append(friend)

        print(self.friends)

            # create new friend


    def accept_friend(self, friend: User):

        pass

    def remove_friend(self, friend: User):

        pass

    def block_user(self, user: User):

        pass

"""
when a user sends a friendship request:

    exsting_relationship = query DB for a record with the ID pair in either column

    if existing_relationship exists:

        if existing_relaitionship is friends or blocked:

            do nothing            
    
    create a record with user's ID as sender ID etc.

when a user accepts a friendship request:

    find existing_relationship

    update enum

    save

when a recipient deletes a friend request:
    
    query by recipient_id and delete the record

to get sent friendship requests:

    use user ID to query sender_id column where enum is pending

to get received friendship requests:

    use user ID to query sender_id column where enum is pending

to get friends:
    query sender_id by user ID and recipient ID by user ID, where enum is accepted

to block a user:

    query for any records with user ID in sender ID or recipient ID column

    delete these records

    create new record with blocking user as sender ID and blocked user and recipient, with enum as blocked
"""
