import { HttpException } from "@nestjs/common";
import { DeleteResult, EntityManager, InsertResult } from "typeorm";
import { VerificationCode } from "../entities/verification-code.entity";

export class VerificationCodeRepository{
    public async saveVerificationCode(verificationCode: VerificationCode, manager: EntityManager): Promise<InsertResult>{
        try{
            return await manager.createQueryBuilder(VerificationCode, "verificationCode")
                                .insert()
                                .into(VerificationCode).values(verificationCode)
                                .execute(); 
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async findVerificationCodeByEmail(email: string, manager: EntityManager): Promise<VerificationCode>{
        try{
            return await manager.createQueryBuilder(VerificationCode, "verificationCode")
                                .where("email = :email", {email})
                                .orderBy('created_date','DESC')
                                .getOne();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }

    public async deleteVerificationCodeByEmail(email: string, manager: EntityManager): Promise<DeleteResult>{
        try{
            return await manager.createQueryBuilder(VerificationCode, "verificationCode")
                                .delete()
                                .from(VerificationCode)
                                .where("email = :email", {email})
                                .execute();
        }catch(err){
            console.log(err)
            throw new HttpException(err.detail, 500);
        }
    }
}