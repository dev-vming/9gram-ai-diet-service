import { EntityManager, InsertResult } from "typeorm";
import { User } from "../entities/user.entity";
import { HealthInfo } from "../entities/health-info.entity";
import { HttpException } from "@nestjs/common";

export class UserRepository{

    // Create
    public async saveUser(user: User, manager: EntityManager): Promise<InsertResult>{
        try{
            return await manager.createQueryBuilder(User, "user").insert()
                                .into(User,['userId','email','password','providerId','username']).values(user).execute(); 
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    // Read
    public async findUserByEmail(email: string, manager: EntityManager): Promise<User>{
        try{
            return await manager.createQueryBuilder(User, "user").where("email = :email", {email}).getOne();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async findUserInfosByUserId(userId: string, manager: EntityManager){
        try{
            return await manager.query('SELECT * FROM "user" AS u LEFT JOIN health_info as h ON u.user_id = h.user_id WHERE u.user_id = $1 ORDER BY h.updated_date DESC limit 1;', [userId]);
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async findUserByUserId(userId: string, manager: EntityManager): Promise<User>{
        try{
            return await manager.createQueryBuilder(User, "user").select().where("user_id = :userId",{userId}).getOne();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async findUserByUserName(username: string, manager: EntityManager): Promise<User>{
        try{
            return await manager.createQueryBuilder(User, "user").select().where("username = :username",{username}).getOne();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    // Update
    public async updateUserByUserId(userId: string, user: User, manager: EntityManager){
        try{
            // console.log("업데이트 될것", user);
            return await manager.createQueryBuilder(User, "user").update(User).set(user).where("user_id = :userId",{userId}).execute();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async updateRecentHealthInfoIdByUserId(userId: string, recentHealthInfoId: string, manager: EntityManager){
        try{
            return await manager.createQueryBuilder(User, "user").update(User).set({recentHealthInfoId}).where("user_id = :userId",{userId}).execute();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    // soft Delete
    public async deleteUserByUserId(userId: string, manager: EntityManager){       
        try{
            await manager.createQueryBuilder(User, "user").update({email:userId, username: userId, providerId: null}).where("user_id = :userId",{userId}).execute();
            return await manager.createQueryBuilder(User, "user").softDelete().where("user_id = :userId",{userId}).execute();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

}