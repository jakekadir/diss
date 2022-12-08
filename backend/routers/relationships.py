from schemas import UserInDB, RelationshipType
from fastapi import APIRouter, HTTPException, status, Depends, Form
from sqlalchemy.orm import Session
from models import User, UserRelationship
from dependencies import get_current_active_user, get_db
import crud

router = APIRouter()

@router.post("/send-friend-request", status_code=201)
async def send_friend_request(db: Session = Depends(get_db), 
                current_user: UserInDB = Depends(get_current_active_user),
                friend_username: str = Form()):

    if friend_username == current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users cannot send friendship requests to themselves.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # get friend
    friend = crud.get_users(db, username=friend_username, first=True)

    # if friend can't be found
    if not friend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find user {friend_username}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # check if user has an existing relationship
    existing_relationship = crud.get_relationships(db, user_id=current_user.id, friend_id=friend.id, first=True)

    """
    check if there is no existing relationship because:
        - if the current user has blocked the friend, don't override the block
        - if the current user has already sent a pending request, don't change anything
        - if the current user is already friends with the friend, don't change anything
    """
    if not existing_relationship:


        # check if friend has an existing relationship with the current user
        friend_relationship: UserRelationship = crud.get_relationships(db, user_id=friend.id, friend_id=current_user.id, first=True)

        # if the friend has a relationship with the current user
        if friend_relationship:

            # if friend has blocked the current user
            if friend_relationship.relationship_status == RelationshipType.BLOCKED:
                
                # do nothing
                return

            # if friend has sent current user a friend request
            elif friend_relationship.relationship_status == RelationshipType.PENDING or friend_relationship.relationship_status == RelationshipType.ACCEPTED:
                
                # update friend's relationship to be accepted
                friend_relationship.relationship_status = RelationshipType.ACCEPTED

                # update current_user->friend relationship
                user_relationship_type: RelationshipType= RelationshipType.ACCEPTED
        
        # if there is no existing relationship, the user->friend relationship is pending
        else:
            user_relationship_type: RelationshipType= RelationshipType.PENDING
        
        # create new relationship
        user_relationship = UserRelationship(user_id=current_user.id, friend_id=friend.id, relationship_status=user_relationship_type)
        
        # add new relationship to user
        current_user.relationships.append(user_relationship)

        db.commit() 

        db.refresh(user_relationship)
        # return relationship object?
        return {
            "UserRelationship" : user_relationship
        }

    # error if relationship already exists
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"User already has a relationship with user {friend_username}",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/accept-friend-request")
async def accept_friend_request(db: Session=Depends(get_db),
                                current_user: UserInDB = Depends(get_current_active_user),
                                friend_username: str = Form()):
    
    # get friend
    friend = crud.get_users(db, username=friend_username, first=True)

    # raise error if friend can't be found
    if not friend:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not find user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # search for user relationship from friend->current user
    friend_user_relationship = crud.get_relationships(db, user_id=friend.id, friend_id=current_user.id, first=True)

    # if no friend->user relationship found
    if not friend_user_relationship:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not find relationship with user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # ensure relationship status is pending
    if friend_user_relationship.relationship_status != RelationshipType.PENDING:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending friend request from {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get current user's relationship with this friend
    user_friend_relationship = crud.get_relationships(db, user_id=current_user.id, friend_id=friend.id, first=True)

    # if the current user has an existing relationship with the friend
    if user_friend_relationship:
        
        # if current user has blocked friend
        if user_friend_relationship.relationship_status == RelationshipType.BLOCKED:
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=f"User has blocked {friend_username} so friendship request cannot be accepted.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # if current user has also sent a request - shouldn't be triggered
        if user_friend_relationship.relationship_status == RelationshipType.PENDING or user_friend_relationship.relationship_status == RelationshipType.ACCEPTED:
            
            # accept on both accounts
            user_friend_relationship.relationship_status = RelationshipType.ACCEPTED
            friend_user_relationship.relationship_status = RelationshipType.ACCEPTED

    # create new relationship with friend
    else:

        user_friend_relationship = UserRelationship(user_id=current_user.id, friend_id=friend.id, relationship_status=RelationshipType.ACCEPTED)
        current_user.relationships.append(user_friend_relationship)

        friend_user_relationship.relationship_status = RelationshipType.ACCEPTED
    
    db.commit()
    db.refresh(user_friend_relationship)
    return {"UserRelationship": user_friend_relationship}

@router.post("/delete-friend")
async def delete_friend(db: Session=Depends(get_db),
                                current_user: UserInDB = Depends(get_current_active_user),
                                friend_username: str = Form()):

    # get friend
    friend = crud.get_users(db, username=friend_username, first=True)

    # raise error if friend can't be found
    if not friend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get user->friend relationship
    user_friend_relationship = crud.get_relationships(db, user_id=current_user.id, friend_id=friend.id, first=True)

    if not user_friend_relationship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find relationship with user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # if not friends with the named user
    if user_friend_relationship.relationship_status != RelationshipType.ACCEPTED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not find friendship with user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        ) 
    
    # remove from relationships array
    current_user.relationships.remove(user_friend_relationship)

    # get friend->user relationship
    friend_user_relationship = crud.get_relationships(db, user_id=friend.id, friend_id=current_user.id, first=True)
    
    # don't override the friend->user block if it exists
    if friend_user_relationship.relationship_status != RelationshipType.BLOCKED:
        
        friend.relationships.remove(friend_user_relationship)
    
    db.commit()
    return f"Friendship with {friend.username} removed."

@router.post("/deny-friend-request")
async def deny_friend_request(db: Session=Depends(get_db), 
                            current_user: UserInDB = Depends(get_current_active_user),
                            friend_username: str = Form()):
    """
    Denies a friend request sent to the user from a named user  . If the friend doesn't exist or hasn't sent the user a friend request, nothing will happen.
    """
    # get friend
    friend = crud.get_users(db, username=friend_username, first=True)

    if not friend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    friend_user_relationship = friend.relationships.filter(UserRelationship.friend_id==current_user.id).first()

    if not friend_user_relationship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find a friend request from user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif friend_user_relationship.relationship_status != RelationshipType.PENDING:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find a friend request from user {friend_username}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # delete friendship request
    friend.relationships.remove(friend_user_relationship)

    db.commit()

@router.post("/block-user")
async def block_user(db: Session=Depends(get_db),
                        current_user: UserInDB = Depends(get_current_active_user),
                        username_to_block: str = Form()):

    user_to_block: User = crud.get_users(db, username=username_to_block, first=True)

    if not user_to_block:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find user {username_to_block}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # get existing relationship
    user_blocking_relationship = current_user.relationships.filter(UserRelationship.user_id==current_user.id, UserRelationship.friend_id==user_to_block.id).first()

    # if the user has no relationship with the user to block
    if not user_blocking_relationship:
        # create a relationship
        user_blocking_relationship = UserRelationship(user_id=current_user.id, friend_id=user_to_block.id, relationship_status=RelationshipType.BLOCKED)

        current_user.relationships.append(user_blocking_relationship)

    else:
        # otherwise update the existing relationship
        user_blocking_relationship.relationship_status = RelationshipType.BLOCKED

    # find a relationship from the user_to_block->current user
    blocking_user_relationship = user_to_block.relationships.filter(UserRelationship.user_id==user_to_block.id, UserRelationship.friend_id==current_user.id).first()

    if blocking_user_relationship:

        user_to_block.relationships.remove(blocking_user_relationship)

    db.commit()
    db.refresh(user_blocking_relationship)
    return {
        "UserRelationship" : user_blocking_relationship
    }

@router.post("/unblock-user")
async def unblock_user(db: Session=Depends(get_db),
                        current_user: UserInDB = Depends(get_current_active_user),
                        username_to_unblock: str = Form()):

    user_to_unblock: User = crud.get_users(db, username=username_to_unblock, first=True)

    if not user_to_unblock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find user {username_to_unblock}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # get existing relationship
    user_unblocking_relationship = current_user.relationships.filter(UserRelationship.user_id==current_user.id, UserRelationship.friend_id==user_to_unblock.id).first()

    # if the user has no relationship with the user to unblock
    if not user_unblocking_relationship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find a relationship with user {username_to_unblock}.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # remove the blocking relationship
    current_user.relationships.remove(user_unblocking_relationship)
    db.commit()

    return {
        "Detail" : f"Unblocked user {username_to_unblock}."
    }